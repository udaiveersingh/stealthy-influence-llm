"""
Which free models are fast and clean enough to use as the second model? Compare several in a few minutes.

For each model: 3 sequential 'POST:' requests (the pipeline's shape), models tested in parallel with each other.
Reports mean seconds per call, how many replies were usable (non-empty, finished normally, starts with POST, no <think>
leak), and how much hidden reasoning came back. Nothing is retried. Request timeout is 150 s.

  python probe_latency.py
  python probe_latency.py --models z-ai/glm-5-3-flash deepseek-ai/deepseek-v4.1-flash

Needs NVIDIA_API_KEY in this window. ~12 tiny requests in total. The default list comes from the free-endpoint page at
build.nvidia.com; if a name is wrong the error is printed, copy the exact id from the model card.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import openai

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODELS = ["z-ai/glm-5-3-flash", "deepseek-ai/deepseek-v4.1-flash", "meta/muse-glimmer-30b", "z-ai/glm-5-3"]
SYSTEM = "You are an ordinary participant in a group discussion. Reply briefly."
POST = ("Discussion topic: should governments heavily regulate advanced AI?\n\nWrite your next message: 1-2 short sentences.\n\n"
        "Respond in EXACTLY this format, with nothing before or after it:\nPOST: <your message>")
results = {}


def worker(model, reps):
    client = openai.OpenAI(base_url=BASE_URL, api_key=os.environ["NVIDIA_API_KEY"], max_retries=0, timeout=150)
    rows = []
    for _ in range(reps):
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=model, max_tokens=400, temperature=0.9,
                                               messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": POST}])
            m = r.choices[0].message
            text = (m.content or "").strip()
            reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or ""
            ok = bool(text) and r.choices[0].finish_reason == "stop" and text[:12].upper().find("POST") >= 0 and "<think" not in text.lower()
            rows.append((time.time() - t0, ok, len(reasoning), text[:110], None))
        except Exception as e:  # noqa: BLE001
            rows.append((time.time() - t0, False, 0, "", f"{type(e).__name__}: {str(e)[:120]}"))
        time.sleep(1)
    results[model] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if not os.environ.get("NVIDIA_API_KEY"):
        sys.exit("NVIDIA_API_KEY is not set in this window.")
    ts = [threading.Thread(target=worker, args=(m, args.reps)) for m in args.models]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print("=" * 90)
    for m in args.models:
        rows = results.get(m, [])
        good = sum(1 for r in rows if r[1])
        mean = sum(r[0] for r in rows) / max(len(rows), 1)
        rs = sum(r[2] for r in rows) / max(len(rows), 1)
        print(f"{m:<38} usable {good}/{len(rows)}   mean {mean:6.1f}s/call   hidden reasoning ~{rs:.0f} chars")
        for r in rows:
            print(f"     {r[0]:6.1f}s  {'ok ' if r[1] else 'BAD'}  {r[4] or repr(r[3])}")
    print("=" * 90)
    print("Want: usable 3/3 and a few seconds per call. Anything near 100+ s/call cannot run this study.")


if __name__ == "__main__":
    main()