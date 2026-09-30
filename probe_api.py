"""
Find out what the NVIDIA endpoint will actually accept, instead of guessing.

Why this exists: two full runs died in a wall of 429s even after the client was
limited to 20 requests/minute. A rate limit set from a guess can't be trusted
when the guess has already been wrong twice. This sends a handful of tiny
requests and reports what the server does.

  Stage 0  ONE isolated request. If even this is rejected, the problem is not
           your request rate (see the message it prints) and nothing client-side
           will fix it. It stops there.
  Stage 1+ A ramp: 4, 8, 15, 25, 40 requests/minute, one thread, a few requests
           per step. Stops at the first step that gets 2+ rejections.
  Burst    8 requests at the same instant, to see how bursts are treated.

Costs about 50 tiny requests and ~3-4 minutes. It bypasses llm_client.py on
purpose (max_retries=0, no throttling) so it shows the raw server behaviour.

If you have just had a storm of 429s, wait ~10 minutes first: continued
requests may prolong a throttle (that timing is a guess, not something I could
verify). Use --wait 600 to have the script do the waiting.

Usage:
    python probe_api.py
    python probe_api.py --wait 600
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import openai

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
STAGES = [(4, 5), (8, 6), (15, 8), (25, 10), (40, 12)]   # (requests/min, requests sent)
BURST_SIZE = 8
TOTAL_CALLS_FOR_RUN = 4200


def one_request(client, model):
    """Returns (ok, status, headers, seconds, detail). Never raises."""
    t0 = time.time()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model, max_tokens=16, temperature=0,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return True, 200, dict(raw.headers), time.time() - t0, ""
    except openai.RateLimitError as e:
        headers = dict(e.response.headers) if getattr(e, "response", None) is not None else {}
        return False, 429, headers, time.time() - t0, str(getattr(e, "body", None) or e)[:160]
    except Exception as e:  # auth, connection, 5xx ...
        return False, type(e).__name__, {}, time.time() - t0, str(e)[:160]


def interesting(headers):
    keys = ("ratelimit", "retry-after", "remaining", "reset", "limit")
    return {k: v for k, v in headers.items() if any(w in k.lower() for w in keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL))
    ap.add_argument("--wait", type=float, default=0, help="seconds to rest before probing")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="testing only: divides the spacing between requests")
    args = ap.parse_args()

    key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY is not set."); sys.exit(1)
    client = openai.OpenAI(base_url=args.base_url, api_key=key, max_retries=0)

    print("=" * 78)
    print(f"API PROBE -- {args.model}")
    print("=" * 78)
    if args.wait:
        print(f"resting {args.wait:.0f}s first ...")
        time.sleep(args.wait)

    # ---- stage 0: one isolated request ----
    ok, status, headers, secs, detail = one_request(client, args.model)
    print(f"\nStage 0 -- a single isolated request: {'ACCEPTED' if ok else 'REJECTED'} "
          f"(status {status}, {secs:.1f}s)")
    info = interesting(headers)
    print("  rate-limit headers:", info if info else "none returned")
    if not ok:
        print(f"  server said: {detail}")
        if status == 429:
            print("\n  A lone request was refused, so this is NOT about how fast you send.")
            print("  Likely causes, in order of my suspicion:")
            print("   1. A throttle window left over from the recent storms -> rest 10+ minutes")
            print("      (python probe_api.py --wait 600) and try again.")
            print("   2. The key's quota/credits are used up -> check your usage page on")
            print("      build.nvidia.com, or try a fresh key.")
            print("   3. NVIDIA-side overload -> nothing you can change; try again later.")
            print("   Also check for a stray python still running:  Get-Process python")
        else:
            print("  Not a rate-limit problem (auth / connection?). Fix that first.")
        print("\n  Do NOT launch the seed run yet.")
        sys.exit(2)

    # ---- ramp ----
    print("\nRamp (one thread, evenly spaced):")
    print(f"  {'rate':>10}  {'sent':>5}  {'ok':>4}  {'429s':>5}  {'avg latency':>12}")
    last_clean, stopped_at = None, None
    for rpm, n in STAGES:
        spacing = 60.0 / rpm / args.time_scale
        results, t_start = [], time.monotonic()
        for i in range(n):
            wait = t_start + i * spacing - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            results.append(one_request(client, args.model))
        n_ok = sum(1 for r in results if r[0])
        n429 = sum(1 for r in results if r[1] == 429)
        n_other = len(results) - n_ok - n429
        lat = sum(r[3] for r in results) / len(results)
        extra = f"  (+{n_other} non-429 errors)" if n_other else ""
        print(f"  {rpm:>6}/min  {n:>5}  {n_ok:>4}  {n429:>5}  {lat:>10.1f}s{extra}")
        if n429 >= 2 or n_other:
            stopped_at = rpm
            break
        last_clean = rpm

    # ---- burst ----
    burst_ok = None
    if last_clean and last_clean >= 8:
        # Rest first: a burst measured right after a saturated step would be
        # rejected because the window is still full, which says nothing about
        # burst tolerance. (A full rate-limit window is assumed to be ~60s.)
        print("\nresting 60s so the burst isn't judged against a still-full window ...")
        time.sleep(60.0 / args.time_scale)
        out, threads = [], []
        def worker():
            out.append(one_request(client, args.model))
        for _ in range(BURST_SIZE):
            t = threading.Thread(target=worker); threads.append(t)
        for t in threads: t.start()
        for t in threads: t.join()
        burst_ok = sum(1 for r in out if r[0])
        print(f"\nBurst -- {BURST_SIZE} requests at the same instant: {burst_ok} accepted, "
              f"{sum(1 for r in out if r[1] == 429)} rejected")

    # ---- recommendation ----
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    if last_clean is None:
        print("Even 4 requests/minute was not clean. Rest longer and re-run the probe; if it")
        print("repeats, suspect quota or NVIDIA-side load rather than your settings.")
        sys.exit(3)
    if stopped_at:
        rpm_rec = max(1, math.floor(0.7 * last_clean))
        print(f"Clean up to {last_clean}/min; rejections began at {stopped_at}/min.")
        print(f"Recommended, with headroom (70% of the last clean rate): {rpm_rec}/min")
    else:
        rpm_rec = last_clean
        print(f"No rejections up to {last_clean}/min (the top of this ramp). Your limit is at")
        print(f"least that; {rpm_rec}/min is safe. I did not probe higher.")
    conc = 8 if burst_ok in (None, BURST_SIZE) else max(1, burst_ok // 2)
    print()
    print(f'  $env:LLM_MAX_RPM="{rpm_rec}"')
    print(f'  $env:LLM_MAX_CONCURRENCY="{conc}"')
    hours = TOTAL_CALLS_FOR_RUN / rpm_rec / 60
    print(f"\nAt {rpm_rec}/min the ~{TOTAL_CALLS_FOR_RUN} calls of the full seed run need at least "
          f"{hours:.1f} hours of rate-limited time.")
    if hours > 12:
        print("That is long. Consider fewer seeds or fewer mechanisms per seed rather than pushing")
        print("the rate and repeating the storms.")


if __name__ == "__main__":
    main()