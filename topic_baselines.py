"""
Which topics leave room for an attacker to push in EITHER direction? Free, no API calls.

For each topic, from the organic (no-attacker) trials already on disk: the mean starting stance, how much
groups drift on their own, and the headroom toward each end of the scale. A topic whose agents start near 0
lets you test a target of -0.8 and +0.8 symmetrically. ai_regulation starts around +0.55, so a +0.8 target
has almost no headroom: the effect is capped near +0.25 before any attacker does anything.

Reads logs/organic_*.json and logs/confirmatory/organic_*.json (Step 1-5 baseline plus the confirmatory
organic arm). Prints a table sorted by how close the starting mean is to neutral.

Usage:
    python topic_baselines.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict


def main():
    paths = sorted(glob.glob("logs/organic_*.json")) + sorted(glob.glob("logs/confirmatory/organic_*.json"))
    by_topic = defaultdict(list)
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:               # noqa: BLE001
            continue
        if not isinstance(d, dict) or "pre_stances" not in d or "post_stances" not in d or not d.get("topic_id"):
            continue
        pre, post = d["pre_stances"], d["post_stances"]
        ids = [a for a in pre if a in post]
        if len(ids) < 5:
            continue
        by_topic[d["topic_id"]].append((st.mean(pre[a] for a in ids), st.mean(post[a] for a in ids),
                                        st.mean(abs(pre[a] - st.mean(pre[b] for b in ids)) for a in ids)))
    if not by_topic:
        print("No organic logs found (expected logs/organic_*.json).")
        return

    rows = []
    for topic, v in by_topic.items():
        pre_m = st.mean(x[0] for x in v)
        rows.append({"topic": topic, "n": len(v), "pre": pre_m, "pre_sd": st.stdev([x[0] for x in v]) if len(v) > 1 else float("nan"),
                     "drift": st.mean(x[1] - x[0] for x in v), "spread": st.mean(x[2] for x in v),
                     "room_neg": pre_m - (-0.8), "room_pos": 0.8 - pre_m})
    rows.sort(key=lambda r: abs(r["pre"]))

    print("=" * 100)
    print("ORGANIC BASELINES BY TOPIC (no attackers)")
    print("=" * 100)
    print(f"  {'topic':<26}{'trials':>7}{'start mean':>12}{'sd across':>11}{'drift (3 rds)':>15}{'within-group':>14}{'room to -0.8':>14}{'room to +0.8':>14}")
    for r in rows:
        print(f"  {r['topic']:<26}{r['n']:>7}{r['pre']:>+12.2f}{r['pre_sd']:>11.2f}{r['drift']:>+15.2f}{r['spread']:>14.2f}"
              f"{r['room_neg']:>14.2f}{r['room_pos']:>14.2f}")
    good = [r for r in rows if abs(r["pre"]) < 0.30 and r["n"] >= 3]
    print("\n  'within-group' = average distance of an agent's starting stance from its group's mean (how split the group is).")
    print("  'room' = distance from the starting mean to that end of the scale.")
    if good:
        print("\n  Topics starting within 0.30 of neutral (room to push either way): " + ", ".join(r["topic"] for r in good))
        print("  These are the candidates for a direction-flip test (target -0.8 AND +0.8) and for generalisation.")
    else:
        print("\n  No topic starts within 0.30 of neutral in this data; a symmetric flip test will be headroom-limited.")


if __name__ == "__main__":
    main()