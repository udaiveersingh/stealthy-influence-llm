"""
What do we already know about the high-status agent (A7_STATUS)? Free, no API calls.

IMPORTANT LIMIT: A7 has never been an attacker in any run, so existing data cannot say whether a high-status
ATTACKER persuades more. That needs the experiment (A7 as the attacker, against an ordinary attacker on the same
seeds). What the data CAN say:

  A. SUSCEPTIBILITY. In the attack arms, is A7 moved more or less than the other measured agents, relative to the
     organic arm of the same seed? (A7 is always a measured agent.) Paired by seed.

  B. PULL IN ORDINARY CONVERSATION. In the organic arm, do the other agents move toward an agent's initial opinion?
     For each agent X: slope of (mean shift of the others) on (X's initial stance minus the others' mean initial
     stance), across seeds. NOTE: every agent's slope is inflated by a shared regression-to-the-centre artifact, so
     compare A7's slope with the OTHER agents' slopes, never with zero.

Both are exploratory observations from one topic. Neither is evidence that status 'works'.

Usage:
    python status_check.py
    python status_check.py --manifest runs/confirmatory_enf_manifest.json
"""
from __future__ import annotations

import argparse
import statistics as st

import analyze_confirmatory as AC

STATUS = "A7_STATUS"


def ols_slope(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return float("nan"), float("nan")
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    resid = [c - (my + b * (a - mx)) for a, c in zip(x, y)]
    se = (sum(r * r for r in resid) / max(len(x) - 2, 1) / sxx) ** 0.5
    return b, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/confirmatory_enf_manifest.json")
    args = ap.parse_args()
    cells = AC.load_cells(args.manifest)
    seeds = sorted(s for s, c in cells.items() if "organic" in c)
    if len(seeds) < 5:
        print("Need at least 5 seeds with an organic arm.")
        return
    meta0 = cells[seeds[0]]["organic"]["meta"]
    sign = 1 if meta0["target_stance"] >= 0 else -1

    print("=" * 96)
    print("A. IS THE HIGH-STATUS AGENT MORE OR LESS PERSUADABLE THAN ITS PEERS?  (attack arm minus organic, toward target)")
    print("=" * 96)
    print("  difference = A7's effect minus the mean effect of the other measured agents; positive = A7 moved MORE")
    print(AC.HEADER)
    for arm in ("attack1", "attack4"):
        for field, label in (("post_stances", "end of attack"), ("post_removal_stances", "final")):
            d = []
            for s in seeds:
                c, o = cells[s].get(arm), cells[s]["organic"]
                if not c:
                    continue
                measured = c["meta"]["measured_ids"]
                if STATUS not in measured:
                    continue
                others = [a for a in measured if a != STATUS]

                def eff(trial, ids):
                    return AC.shift(trial, ids, field, sign)
                a7 = eff(c["trial"], [STATUS]) - eff(o["trial"], [STATUS])
                oth = eff(c["trial"], others) - eff(o["trial"], others)
                d.append(a7 - oth)
            if len(d) >= 3:
                print(AC.row(f"{arm}, {label}", AC.paired_summary(d)))

    print("\n" + "=" * 96)
    print("B. IN ORDINARY CONVERSATION, DO OTHERS MOVE TOWARD AN AGENT'S STARTING OPINION?  (organic arm only)")
    print("=" * 96)
    agents = sorted(cells[seeds[0]]["organic"]["trial"]["pre_stances"])
    for field, label in (("post_stances", "after round 3"), ("post_removal_stances", "after round 6")):
        slopes = {}
        for x in agents:
            gaps, shifts = [], []
            for s in seeds:
                t = cells[s]["organic"]["trial"]
                pre, post = t["pre_stances"], t[field]
                others = [a for a in pre if a != x]
                gaps.append(pre[x] - st.mean(pre[o] for o in others))
                shifts.append(st.mean(post[o] - pre[o] for o in others))
            slopes[x] = ols_slope(gaps, shifts)
        ranked = sorted(slopes, key=lambda a: slopes[a][0], reverse=True)
        peers = [slopes[a][0] for a in agents if a != STATUS]
        print(f"\n  {label}: slope of others' shift on the agent's initial gap (n = {len(seeds)} seeds)")
        for a in ranked:
            b, se = slopes[a]
            mark = "   <-- high status" if a == STATUS else ""
            print(f"    {a:<14} {b:+.3f}  (se {se:.3f})  rank {ranked.index(a) + 1}/{len(agents)}{mark}")
        print(f"    A7 minus the mean of its peers: {slopes[STATUS][0] - st.mean(peers):+.3f}  "
              f"(peers range {min(peers):+.3f} to {max(peers):+.3f})")

    print("\nREADING")
    print("  - A: a clearly positive difference means the high-status agent is easier to move; negative means harder.")
    print("  - B: if A7's slope is not clearly above its peers' (it will usually sit inside their range with this n),")
    print("    there is no sign in ordinary conversation that status gives an agent extra pull.")
    print("  - Neither says whether a high-status ATTACKER persuades more. Only the attacker experiment can.")


if __name__ == "__main__":
    main()