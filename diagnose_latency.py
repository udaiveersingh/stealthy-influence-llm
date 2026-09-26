"""
Diagnose the API latency slowdown. Costs ~10 real calls (cheap, ~1 min if
the API is healthy, longer if it isn't -- which is itself the answer).

Your logs show old organic trials at ~2s/call and the new run at ~80-120s/call,
with the SAME provider, model, agent count and condition. The slowdown is not
in your simulation code. This script narrows down what it actually is.

Run:  python diagnose_latency.py
"""
import statistics
import time

from src.llm_client import LLMClient

SYSTEM = "You are an ordinary participant in a group discussion. Reply briefly."
USER = (
    "Discussion topic: should governments heavily regulate AI?\n\n"
    "Write your next message: 1-2 short sentences.\n\n"
    "Respond in EXACTLY this format, with nothing before or after it:\n"
    "POST: <your message>"
)


def timed_call(client, max_tokens, label):
    t0 = time.time()
    try:
        raw = client.complete(SYSTEM, USER, temperature=0.9, max_tokens=max_tokens)
        elapsed = time.time() - t0
        chars = len(raw)
        print(f"  {label:<22}{elapsed:>8.1f}s   {chars:>5} chars returned")
        return elapsed, chars
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {label:<22}{elapsed:>8.1f}s   ERROR: {type(e).__name__}: {e}")
        return elapsed, None


def main():
    client = LLMClient(provider="nvidia")
    info = client.info()
    print("=" * 70)
    print("API LATENCY DIAGNOSTIC")
    print("=" * 70)
    print(f"provider={info['provider']}  model={info['model']}")
    print()

    print("A) Five identical calls at your production max_tokens=1800:")
    prod = [timed_call(client, 1800, f"call {i+1}") for i in range(5)]
    prod_times = [t for t, _ in prod]

    print()
    print("B) Same prompt at much lower max_tokens (tests whether generation")
    print("   length is the bottleneck):")
    low = [timed_call(client, 200, f"max_tokens=200 #{i+1}") for i in range(3)]
    low_times = [t for t, _ in low]

    print()
    print("=" * 70)
    print("READING THE RESULT")
    print("=" * 70)
    med_prod = statistics.median(prod_times)
    med_low = statistics.median(low_times)
    print(f"  median @ max_tokens=1800: {med_prod:>7.1f}s")
    print(f"  median @ max_tokens=200:  {med_low:>7.1f}s")
    print(f"  your OLD baseline runs:       ~2.1s")
    print(f"  your NEW Step 6 run:         ~80-120s")
    print()

    if med_prod < 10:
        print("  -> API is FAST right now. The slowdown during your Step 6 run was")
        print("     TRANSIENT (time-of-day load / temporary throttling), not a")
        print("     permanent change. Re-run overnight and it may well be fine.")
        print("     Start with --seeds 1 and watch the first trial's pace before")
        print("     committing to the full run.")
    elif med_low < med_prod / 3:
        print("  -> Lowering max_tokens helped a lot: generation length is the")
        print("     bottleneck. The model is emitting long output before stopping.")
        print("     Consider a middle value (e.g. 600-800) -- but note your")
        print("     max_tokens=1800 exists because of real truncation bugs, so")
        print("     lowering it risks reintroducing those. Test before committing.")
    else:
        print("  -> Still slow regardless of max_tokens, and slow right now. This")
        print("     points at ACCOUNT-LEVEL RATE LIMITING rather than your prompts.")
        print("     Check: (a) did you rotate your NVIDIA API key recently? A new")
        print("     free-tier key can sit on different limits than the one your old")
        print("     fast baseline used. (b) Does src/llm_client.py silently retry")
        print("     with backoff on HTTP 429? If so, one 'call' may be several")
        print("     attempts plus sleeps -- add a print on retry so you can see it.")
        print("     (c) Check your usage/quota page at build.nvidia.com.")

    spread = max(prod_times) - min(prod_times)
    if spread > med_prod:
        print()
        print(f"  NOTE: wide spread across identical calls ({min(prod_times):.1f}s -"
              f" {max(prod_times):.1f}s) -- consistent with server-side queueing.")


if __name__ == "__main__":
    main()
