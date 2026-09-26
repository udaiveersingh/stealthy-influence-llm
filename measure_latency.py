"""
Measure real per-call API latency from trial logs.

Message.timestamp is wall-clock at the moment the Message object was created,
i.e. just after that agent's completion returned. So the gap between two
consecutive messages in a trial IS the latency of the second call (plus any
retries it needed, plus throttle).

Use this to compare your OLD organic baseline runs against NEW runs and find
out what actually changed -- rather than guessing between max_tokens,
pin_first_round, agent count, retries, or API-side load.

Usage:
    python measure_latency.py logs/*.json
    python measure_latency.py logs/organic_*.json      # just the old baseline
    python measure_latency.py logs/*.json --by-condition
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict


def trial_gaps(trial: dict):
    """Per-call latencies (seconds) between consecutive messages."""
    msgs = sorted(trial.get("messages", []), key=lambda m: m.get("timestamp", 0))
    ts = [m["timestamp"] for m in msgs if m.get("timestamp")]
    return [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--by-condition", action="store_true",
                        help="group the summary by trial condition")
    args = parser.parse_args()

    paths = []
    for p in args.paths:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
    if not paths:
        print("No log files matched.")
        return

    rows = []
    for path in paths:
        try:
            with open(path) as f:
                trial = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        gaps = trial_gaps(trial)
        if len(gaps) < 3:
            continue
        rows.append({
            "path": path,
            "trial_id": trial.get("trial_id", "?")[:8],
            "condition": trial.get("condition", "?"),
            "topic": trial.get("topic_id", "?"),
            "model": trial.get("model_name", "?"),
            "provider": trial.get("model_provider", "?"),
            "n_messages": len(trial.get("messages", [])),
            "median": statistics.median(gaps),
            "mean": statistics.mean(gaps),
            "min": min(gaps),
            "max": max(gaps),
        })

    if not rows:
        print("No trials with enough messages to measure.")
        return

    print("=" * 108)
    print("PER-CALL LATENCY BY TRIAL (seconds between consecutive messages)")
    print("=" * 108)
    print(f"{'trial':<10}{'condition':<26}{'msgs':>5}{'median':>9}{'mean':>9}"
          f"{'min':>8}{'max':>8}  {'provider':<9}")
    print("-" * 108)
    for r in sorted(rows, key=lambda x: x["median"]):
        print(f"{r['trial_id']:<10}{r['condition']:<26}{r['n_messages']:>5}"
              f"{r['median']:>9.1f}{r['mean']:>9.1f}{r['min']:>8.1f}{r['max']:>8.1f}  "
              f"{r['provider']:<9}")

    def summarize(label, subset):
        meds = [r["median"] for r in subset]
        print(f"{label:<30}{len(subset):>5} trials   "
              f"median-of-medians={statistics.median(meds):>7.1f}s   "
              f"fastest={min(meds):>6.1f}s   slowest={max(meds):>7.1f}s")

    print("\n" + "=" * 108)
    print("SUMMARY")
    print("=" * 108)
    summarize("ALL TRIALS", rows)

    by_provider = defaultdict(list)
    for r in rows:
        by_provider[r["provider"]].append(r)
    if len(by_provider) > 1:
        print()
        for prov, subset in sorted(by_provider.items()):
            summarize(f"  provider={prov}", subset)

    if args.by_condition:
        by_cond = defaultdict(list)
        for r in rows:
            by_cond[r["condition"]].append(r)
        print()
        for cond, subset in sorted(by_cond.items()):
            summarize(f"  condition={cond}", subset)

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    if len(by_model) > 1:
        print()
        for model, subset in sorted(by_model.items()):
            summarize(f"  model={model[:24]}", subset)

    print("\n" + "=" * 108)
    print("HOW TO READ THIS")
    print("=" * 108)
    print("  Compare your OLD organic baseline trials against the NEW Step 6 trials.")
    print("  - If old and new are both slow  -> the slowdown is API-side load, not your code.")
    print("  - If old is fast and new is slow -> something in your config changed. Prime")
    print("    suspects, in order: max_tokens (stance.py raised 350->1200; conversation.py")
    print("    uses 1800), pin_first_round=True (new -- roughly doubles prompt history in")
    print("    rounds 2-3), agent count 7->8, and extra retries from the stricter detector.")
    print("  - A huge min/max spread within ONE trial (e.g. 26s to 537s) points at")
    print("    server-side queueing rather than anything about your prompts.")


if __name__ == "__main__":
    main()
