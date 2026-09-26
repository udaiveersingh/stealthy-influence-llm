"""
Does concurrency help?

Your calls are spending minutes WAITING, not generating (~0.25 chars/sec,
tiny outputs). The question that decides your overnight run is whether
several requests can wait *at the same time*, or whether the provider
serializes them anyway.

  - If 4 parallel calls finish in about the time of ONE call
        -> parallelism rescues the run. A 1120-call run drops from ~39h
           to a few hours. Worth wiring into the pipeline.
  - If 4 parallel calls take about as long as 4 sequential calls
        -> you're behind a hard serialized queue / concurrency cap.
           Parallelism won't help; you need a different provider or to
           wait out whatever is throttling this account.

Costs 8 real calls (4 sequential + 4 parallel). Given current latency this
may take 10-20 minutes to finish -- that slowness is itself the finding.

Run:  python test_concurrency.py
"""
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from src.llm_client import LLMClient

SYSTEM = "You are an ordinary participant in a group discussion. Reply briefly."
USER = (
    "Discussion topic: should governments heavily regulate AI?\n\n"
    "Write your next message: 1-2 short sentences.\n\n"
    "Respond in EXACTLY this format, with nothing before or after it:\n"
    "POST: <your message>"
)

N = 4


def one_call(idx):
    """Each thread gets its own client -- don't assume the real client is
    thread-safe for shared use."""
    client = LLMClient(provider="nvidia")
    t0 = time.time()
    try:
        raw = client.complete(SYSTEM, USER, temperature=0.9, max_tokens=400)
        return idx, time.time() - t0, len(raw), None
    except Exception as e:
        return idx, time.time() - t0, None, f"{type(e).__name__}: {e}"


def main():
    print("=" * 70)
    print("CONCURRENCY TEST")
    print("=" * 70)
    print(f"Running {N} sequential calls, then {N} parallel calls.")
    print("This may take a while at current latency -- that's expected.\n")

    # ---- sequential ----
    print(f"A) {N} SEQUENTIAL calls:")
    seq_start = time.time()
    seq_times = []
    for i in range(N):
        _, elapsed, chars, err = one_call(i)
        seq_times.append(elapsed)
        status = f"{chars} chars" if err is None else f"ERROR {err}"
        print(f"   call {i+1}: {elapsed:>7.1f}s   {status}")
    seq_total = time.time() - seq_start
    print(f"   sequential TOTAL: {seq_total:.1f}s")

    # ---- parallel ----
    print(f"\nB) {N} PARALLEL calls:")
    par_start = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        results = list(ex.map(one_call, range(N)))
    par_total = time.time() - par_start
    for idx, elapsed, chars, err in sorted(results):
        status = f"{chars} chars" if err is None else f"ERROR {err}"
        print(f"   call {idx+1}: {elapsed:>7.1f}s   {status}")
    print(f"   parallel TOTAL:   {par_total:.1f}s")

    # ---- verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    speedup = seq_total / par_total if par_total > 0 else 0
    print(f"  sequential total: {seq_total:>8.1f}s")
    print(f"  parallel total:   {par_total:>8.1f}s")
    print(f"  speedup:          {speedup:>8.1f}x   (ideal would be ~{N}x)")
    print()

    errors = [r for r in results if r[3] is not None]
    if errors:
        print(f"  WARNING: {len(errors)}/{N} parallel calls ERRORED. If these are")
        print("  rate-limit (429) errors, the provider is rejecting concurrency --")
        print("  parallelism will not help and may make things worse.")
        print()

    if speedup >= N * 0.6:
        print("  -> CONCURRENCY WORKS. Parallelising is the fix.")
        print(f"     Your 1120-call run would drop from ~39h to roughly")
        print(f"     {39/speedup:.1f}h at this speedup.")
        print("     Best targets, in order:")
        print("       1. Stance elicitation (32 of 56 calls per trial, and all")
        print("          8 agents are independent -- safe to parallelise).")
        print("       2. Trials themselves (different seeds are independent).")
        print("     NOT parallelisable: agents within a round -- each one must")
        print("     see the previous agent's message, that's the whole design.")
    elif speedup >= 1.5:
        print("  -> PARTIAL benefit. Some concurrency allowed but capped.")
        print("     Worth parallelising stance elicitation, but test the worker")
        print("     count -- more workers may just trigger rate limiting.")
    else:
        print("  -> CONCURRENCY DOES NOT HELP. You're behind a serialized queue")
        print("     or a hard concurrency cap. Parallelising the pipeline would")
        print("     be wasted work. Realistic options instead:")
        print("       (a) Check credits/quota at build.nvidia.com -- free-tier")
        print("           exhaustion often shows up as throttling, not errors.")
        print("       (b) Try a different provider for the WHOLE Step 6 dataset")
        print("           (organic control + adversarial together, so it stays")
        print("           internally consistent). Your old nvidia baseline stays")
        print("           separate earlier work -- do not mix them in analysis.")
        print("       (c) Wait it out and retry -- the run is resumable, so an")
        print("           overnight attempt costs nothing if it's still slow.")


if __name__ == "__main__":
    main()
