"""
Is this model usable for the pipeline? ~14 tiny requests, a few minutes. Run BEFORE any experiment on a new model.

It sends two prompts shaped like the pipeline's (a 1-2 sentence 'POST:' reply and a 'STANCE:' reply) with each of three
request variants for switching reasoning off, and for every reply reports: seconds, finish_reason, completion tokens,
whether the visible text is empty, how much hidden reasoning came back, and whether '<think>' leaked into the text.

  python probe_model.py --model <model name exactly as on build.nvidia.com>

Needs NVIDIA_API_KEY in this window. Nothing is retried, so you see the raw server behaviour.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import openai

BASE_URL = "https://integrate.api.nvidia.com/v1"
SYSTEM = "You are an ordinary participant in a group discussion. Reply briefly."
POST = ("Discussion topic: should governments heavily regulate advanced AI?\n\nWrite your next message: 1-2 short sentences.\n\n"
        "Respond in EXACTLY this format, with nothing before or after it:\nPOST: <your message>")
STANCE = ("Topic: Governments should heavily regulate the development of advanced AI systems, even if it slows down innovation.\n"
          "On a scale from -1 (strongly disagree) to +1 (strongly agree), what is your stance?\n"
          "Respond in EXACTLY this format:\nSTANCE: <number>\nREASON: <under 15 words>")
VARIANTS = {
    "A: nothing extra": None,
    "B: chat_template_kwargs enable_thinking=false": {"chat_template_kwargs": {"enable_thinking": False}},
    "C: thinking disabled (Moonshot-style)": {"thinking": {"type": "disabled"}},
}


def call(client, model, prompt, extra, max_tokens):
    t0 = time.time()
    kw = dict(model=model, max_tokens=max_tokens, temperature=0.9,
              messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    if extra:
        kw["extra_body"] = extra
    try:
        r = client.chat.completions.create(**kw)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:200]}", "secs": time.time() - t0}
    m = r.choices[0].message
    text = (m.content or "").strip()
    reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or ""
    return {"secs": time.time() - t0, "finish": r.choices[0].finish_reason,
            "tokens": getattr(getattr(r, "usage", None), "completion_tokens", None),
            "text": text, "reasoning_chars": len(reasoning), "think_leak": "<think" in text.lower()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("NVIDIA_MODEL"))
    ap.add_argument("--max-tokens", type=int, default=400, help="what the pipeline uses for posts")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key or not args.model:
        sys.exit("Need NVIDIA_API_KEY set in this window and --model <name> (or NVIDIA_MODEL).")
    client = openai.OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=300)
    print(f"model: {args.model}   max_tokens: {args.max_tokens}\n")
    summary = {}
    for vname, extra in VARIANTS.items():
        print(f"--- variant {vname}")
        good = n = 0
        secs = []
        for pname, prompt, marker in (("POST", POST, "POST"), ("STANCE", STANCE, "STANCE")):
            for _ in range(args.reps):
                n += 1
                res = call(client, args.model, prompt, extra, args.max_tokens)
                secs.append(res["secs"])
                if "error" in res:
                    print(f"  {pname:6} ERROR after {res['secs']:.1f}s: {res['error']}")
                    continue
                ok = bool(res["text"]) and res["finish"] == "stop" and marker in res["text"][:12].upper() and not res["think_leak"]
                good += ok
                print(f"  {pname:6} {res['secs']:5.1f}s finish={res['finish']} tokens={res['tokens']} "
                      f"reasoning_chars={res['reasoning_chars']} think_leak={res['think_leak']} usable={ok}")
                print(f"         text: {res['text'][:140]!r}")
                time.sleep(2)
        summary[vname] = (good, n, sum(secs) / len(secs) if secs else 0)
        print()
    print("=" * 70)
    for v, (g, n, avg) in summary.items():
        print(f"  {v:<50} usable {g}/{n}   mean {avg:.1f}s/call")
    print("=" * 70)
    print("Pick a variant with usable == all and a tolerable latency. Paste this whole output back to me.")


if __name__ == "__main__":
    main()