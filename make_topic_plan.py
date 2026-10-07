"""
Choose the attackers' target direction for each topic. Free, no API calls.

RULE, fixed in advance and applied identically to every topic:
    push AGAINST the group's natural lean, so there is room to move.
    mean starting stance > +0.10  ->  target -0.80  (attackers argue the DISAGREE side)
    mean starting stance < -0.10  ->  target +0.80  (attackers argue the AGREE side)
    within 0.10 of neutral        ->  target -0.80  (arbitrary but fixed)

Why this rule: ai_regulation starts near +0.55, and a +0.80 target would leave almost no room, so a null there would
mean nothing. Pushing against the lean also keeps one topic from being easy and another impossible. A consequence worth
stating in the paper: the targets differ in sign across topics, so every effect is reported as movement TOWARD the
assigned target, and the natural lean is part of what the attackers are working against.

Reads organic (no-attacker) logs from the Step 1-5 baseline and the confirmatory organic arm, and writes
config/topic_plan.json, which `run_confirmatory.py --target auto` reads.

Usage:
    python make_topic_plan.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

OUT = "config/topic_plan.json"
MIN_TRIALS = 3
NEUTRAL_BAND = 0.10
TARGET = 0.8


def main():
    paths = sorted(glob.glob("logs/organic_*.json")) + sorted(glob.glob("logs/confirmatory/organic_*.json"))
    by = defaultdict(list)
    used = 0
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:               # noqa: BLE001
            continue
        if not isinstance(d, dict) or not d.get("topic_id") or "pre_stances" not in d:
            continue
        pre = list(d["pre_stances"].values())
        if len(pre) < 5:
            continue
        by[d["topic_id"]].append(st.mean(pre))
        used += 1
    if not by:
        print("No organic logs found (expected logs/organic_*.json).")
        return

    plan = {"rule": f"push against the group's natural lean; |start mean| < {NEUTRAL_BAND} -> -{TARGET}",
            "generated_from_trials": used, "topics": {}}
    print("=" * 84)
    print("TARGET PLAN (rule: push AGAINST the group's natural lean)")
    print("=" * 84)
    print(f"  {'topic':<26}{'organic trials':>15}{'start mean':>12}{'target':>9}  attackers argue")
    for topic in sorted(by, key=lambda t: -abs(st.mean(by[t]))):
        vals = by[topic]
        if len(vals) < MIN_TRIALS:
            print(f"  {topic:<26}{len(vals):>15}   too few organic trials (need {MIN_TRIALS}); skipped")
            continue
        m = st.mean(vals)
        target = -TARGET if m > -NEUTRAL_BAND else TARGET
        if abs(m) <= NEUTRAL_BAND:
            target = -TARGET
        side = "DISAGREE" if target < 0 else "AGREE"
        plan["topics"][topic] = {"mean_pre": round(m, 3), "n_trials": len(vals), "target": target, "side": side}
        print(f"  {topic:<26}{len(vals):>15}{m:>+12.2f}{target:>+9.1f}  the {side} side")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(plan, open(OUT, "w"), indent=2)
    near = [t for t, v in plan["topics"].items() if abs(v["mean_pre"]) <= NEUTRAL_BAND]
    print(f"\n  written to {OUT}")
    if near:
        print(f"  Near-neutral topics (direction chosen by the fixed tie-break, not by headroom): {', '.join(near)}")
    print("  Commit this file BEFORE running the topics, so the targets are fixed before any attacker result exists.")


if __name__ == "__main__":
    main()