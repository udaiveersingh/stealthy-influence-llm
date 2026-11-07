"""
Analysis of the 512-agent society runs (run_society.py).

  python analyze_society.py                      # every topic found under runs/society/
  python analyze_society.py --ns society_hpc     # another namespace

Per topic and attack arm, on the measured agents (never an attacker in any arm), paired with the organic twin:
  end    = shift toward the target at the end of the attack (after round 3), attacked minus organic
  final  = the same after the 3 silent rounds (after round 6); retention = final / end
  by hop = the same, split by network distance to the nearest attacker in THAT arm (1 = directly connected)

Inference. Within one society the agents influence each other, so the per-agent 95% CI (bootstrap over agents) is
optimistic and is labelled descriptive. The confirmatory test is ACROSS TOPICS: each topic is an independent
society, and a one-sample t-test on the per-topic effects (plus how many topics are positive) is the headline.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st

from scipy import stats

ARMS = ["attack2pct", "attack10pct"]


def load(p):
    return json.load(open(p))


def boot_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def hops(adj, sources):
    dist = {s: 0 for s in sources}
    frontier = list(sources)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def diffs(org, att, ids, field, sign):
    out = {}
    for a in ids:
        x, y = org[field].get(a), att[field].get(a)
        if x is not None and y is not None:
            out[a] = sign * (y - x)
    return out


def fmt(xs):
    if len(xs) < 3:
        return f"n={len(xs)}"
    lo, hi = boot_ci(xs)
    return f"{st.fmean(xs):+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(xs)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default=os.environ.get("RUN_NS", "").strip() or "society")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: every seed found")
    ap.add_argument("--quiet", action="store_true", help="skip the per-society detail, print the summaries only")
    args = ap.parse_args()

    manifests = sorted(glob.glob(f"runs/{args.ns}/society_manifest__*.json"))
    if not manifests:
        raise SystemExit(f"no society manifests under runs/{args.ns}/")
    # per[arm][topic][seed] = {"end", "final", "hop1_end", "hop1_final", "hop2_end", "hop2_final"}
    per = {a: {} for a in ARMS}
    print("=" * 78)
    print(f"SOCIETY ANALYSIS  namespace={args.ns}  (shift toward the attackers' side, attacked minus organic twin)")
    print("=" * 78)
    for mp in manifests:
        comp = load(mp)["completed"]
        if not comp:
            continue
        topic = next(iter(comp.values()))["topic_id"]
        seeds = sorted({e["seed"] for e in comp.values()})
        if args.seeds:
            seeds = [s for s in seeds if s in args.seeds]
        for seed in seeds:
            okey = f"organic|{topic}|{seed}"
            if okey not in comp:
                continue
            org = load(comp[okey]["path"])
            if not args.quiet:
                print(f"\n{topic}  (society seed {seed})")
            for arm in ARMS:
                k = f"{arm}|{topic}|{seed}"
                if k not in comp:
                    continue
                att, meta = load(comp[k]["path"]), load(comp[k]["meta_path"])
                soc = load(meta["society_path"])
                sign = -1.0 if meta["target_stance"] < 0 else 1.0
                ids = meta["measured_ids"]
                end = diffs({"s": org["post_stances"]}, {"s": att["post_stances"]}, ids, "s", sign)
                fin = diffs({"s": org["post_removal_stances"]}, {"s": att["post_removal_stances"]}, ids, "s", sign)
                e, f = list(end.values()), list(fin.values())
                me, mf = st.fmean(e), st.fmean(f)
                dist = hops(soc["adj"], meta["attackers"])
                rec = {"end": me, "final": mf}
                for h, test in (("hop1", lambda d: d == 1), ("hop2", lambda d: d == 2)):
                    ee = [v for a, v in end.items() if dist.get(a) is not None and test(dist[a])]
                    ff = [v for a, v in fin.items() if dist.get(a) is not None and test(dist[a])]
                    rec[f"{h}_end"] = st.fmean(ee) if ee else float("nan")
                    rec[f"{h}_final"] = st.fmean(ff) if ff else float("nan")
                per[arm].setdefault(topic, {})[seed] = rec
                if args.quiet:
                    continue
                p_e = stats.ttest_1samp(e, 0).pvalue
                print(f"  {arm:12s} ({len(meta['attackers'])} attackers, memory={meta.get('memory', 'short')})  "
                      f"end {fmt(e)}  p={p_e:.3g}")
                print(f"  {'':12s}  final {fmt(f)}  retained {(mf / me * 100) if abs(me) > 1e-9 else float('nan'):.0f}%")
                for label, test in (("hop 1", lambda d: d == 1), ("hop 2", lambda d: d == 2), ("hop 3+", lambda d: d >= 3)):
                    ee = [v for a, v in end.items() if dist.get(a) is not None and test(dist[a])]
                    ff = [v for a, v in fin.items() if dist.get(a) is not None and test(dist[a])]
                    print(f"  {'':12s}    {label:6s} end {fmt(ee):34s} final {fmt(ff)}")
                fails = sum(1 for lg in meta["attacker_logs"].values() for iv in lg.get("interventions", [])
                            if iv.get("forced_direction_failed"))
                n_iv = sum(len(lg.get("interventions", [])) for lg in meta["attacker_logs"].values())
                print(f"  {'':12s}    attacker messages judged off-direction (kept, flagged): {fails}/{n_iv}")

    def test(xs, label):
        xs = [x for x in xs if x == x]
        if len(xs) >= 2:
            t = stats.ttest_1samp(xs, 0)
            return (f"  {label:<34} mean {st.fmean(xs):+.3f}  positive {sum(x > 0 for x in xs)}/{len(xs)}  "
                    f"t={t.statistic:.2f} p={t.pvalue:.3g}")
        return f"  {label:<34} " + (f"{xs[0]:+.3f} (1 unit)" if xs else "no data")

    print("\n" + "=" * 78)
    print("ACROSS TOPICS (unit = topic; with several seeds a topic's value is the mean over its societies)")
    for arm in ARMS:
        if not per[arm]:
            continue
        n_seeds = sorted({len(v) for v in per[arm].values()})
        print(f"\n  {arm}  ({len(per[arm])} topics; societies per topic: {n_seeds})")
        for field, lab in (("end", "end of attack"), ("final", "after silent rounds"),
                           ("hop1_end", "directly connected, end"), ("hop1_final", "directly connected, after"),
                           ("hop2_end", "two steps away, end"), ("hop2_final", "two steps away, after")):
            vals = [st.fmean(r[field] for r in seeds_.values() if r[field] == r[field]) for seeds_ in per[arm].values()]
            print(test(vals, lab))
        all_seeds = sorted({s for v in per[arm].values() for s in v})
        if len(all_seeds) > 1:
            print("  replication by seed (unit = topic, one seed at a time):")
            for s in all_seeds:
                for field in ("end", "final"):
                    vals = [v[s][field] for v in per[arm].values() if s in v]
                    print(test(vals, f"   seed {s}, {field}"))
    print("=" * 78)


if __name__ == "__main__":
    main()
