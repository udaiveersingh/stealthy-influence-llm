"""
Does the effect generalise across topics?

For each topic: the 4-attacker arm's effect on the measured agents against that topic's own organic arm (paired by seed,
toward the topic's target; targets were chosen by a fixed rule, see make_topic_plan.py). Then the topic estimates are
pooled across topics.

Pooling, and why two numbers:
  RANDOM-EFFECTS (DerSimonian-Laird)  allows the true effect to differ by topic; this is the headline, because topics are
                                      not replicates of one another.
  FIXED-EFFECT                        assumes one common effect; shown for reference.
  tau and I^2                         how much the true effect differs between topics. I^2 near 0 = topics agree.
  sign count                          in how many topics the effect is positive. It makes no distributional assumption.

LIMITS, stated before the data: with only 8 topics the between-topic variance is poorly estimated and the normal-theory
interval is optimistic; read the sign count and the per-topic intervals alongside it. The topic targets differ in sign,
and the natural lean differs, so 'toward the target' is the only comparable scale.

Usage:
    python analyze_topics.py                          # diversified arm, every topic that has a manifest
    python analyze_topics.py --kind short             # the length-matched, non-diversified arm
    python analyze_topics.py --topics remote_work college_value
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st

import analyze_confirmatory as AC

DEFAULT_TOPIC = "ai_regulation"
KINDS = {"long": "runs/confirmatory_enf_manifest.json",
         "short": "runs/confirmatory_short_manifest.json",
         "diverse": "runs/confirmatory_diverse_manifest.json"}
ALL_TOPICS = ["ai_regulation", "remote_work", "college_value", "free_speech_harm", "traditional_family_roles",
              "religion_public_life", "centralized_governance", "individual_vs_collective"]


def path_for(kind, topic):
    base = KINDS[kind]
    return base if topic == DEFAULT_TOPIC else base.replace(".json", f"__{topic}.json")


def pool(estimates):
    """estimates: list of (mean, se). DerSimonian-Laird random effects plus fixed effect."""
    k = len(estimates)
    m = [e[0] for e in estimates]
    v = [max(e[1] ** 2, 1e-12) for e in estimates]
    w = [1 / x for x in v]
    fe = sum(wi * mi for wi, mi in zip(w, m)) / sum(w)
    fe_se = math.sqrt(1 / sum(w))
    q = sum(wi * (mi - fe) ** 2 for wi, mi in zip(w, m))
    c = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = [1 / (vi + tau2) for vi in v]
    re = sum(wi * mi for wi, mi in zip(ws, m)) / sum(ws)
    re_se = math.sqrt(1 / sum(ws))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    return {"k": k, "re": re, "re_se": re_se, "re_lo": re - 1.96 * re_se, "re_hi": re + 1.96 * re_se,
            "fe": fe, "fe_se": fe_se, "tau": math.sqrt(tau2), "i2": i2, "q": q,
            "n_pos": sum(1 for x in m if x > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(KINDS), default="diverse")
    ap.add_argument("--topics", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default="runs/topics_analysis.json")
    ap.add_argument("--fig", default="figures/topics_forest.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    topics = args.topics or [t for t in ALL_TOPICS if os.path.exists(path_for(args.kind, t))]
    rows = {}
    for t in topics:
        p = path_for(args.kind, t)
        if not os.path.exists(p):
            print(f"  (skipping {t}: no manifest at {p})")
            continue
        cells = AC.load_cells(p, set(args.seeds) if args.seeds else None)
        seeds = sorted(s for s, c in cells.items() if "organic" in c and "attack4" in c)
        if len(seeds) < 3:
            print(f"  (skipping {t}: only {len(seeds)} seeds with organic + attack4)")
            continue
        meta = cells[seeds[0]]["organic"]["meta"]
        sign = 1 if meta["target_stance"] >= 0 else -1
        e_end, e_fin, drift, start = [], [], [], []
        for s in seeds:
            measured = cells[s]["organic"]["meta"]["measured_ids"]
            o, a = cells[s]["organic"]["trial"], cells[s]["attack4"]["trial"]
            e_end.append(AC.shift(a, measured, "post_stances", sign) - AC.shift(o, measured, "post_stances", sign))
            e_fin.append(AC.shift(a, measured, "post_removal_stances", sign) - AC.shift(o, measured, "post_removal_stances", sign))
            drift.append(AC.shift(o, measured, "post_removal_stances", sign))
            start.append(st.mean(o["pre_stances"].values()))
        rows[t] = {"target": meta["target_stance"], "n": len(seeds), "start": st.mean(start),
                   "end": AC.paired_summary(e_end), "final": AC.paired_summary(e_fin),
                   "decay": AC.paired_summary([f - e for e, f in zip(e_end, e_fin)]),
                   "organic_drift_final": st.mean(drift)}
    if len(rows) < 2:
        print("Need at least two topics with data to pool. Found:", list(rows))
        return

    print("=" * 104)
    print(f"ACROSS TOPICS -- the 4-attacker '{args.kind}' arm against each topic's own organic arm (POSITIVE = toward that topic's target)")
    print("=" * 104)
    print(f"  {'topic':<26}{'target':>7}{'start':>7}{'seeds':>6}  {'end of attack (95% CI)':<28}{'final (95% CI)':<28}{'organic drift':>14}")
    for t, r in rows.items():
        e, f = r["end"], r["final"]
        print(f"  {t:<26}{r['target']:>+7.1f}{r['start']:>+7.2f}{r['n']:>6}  "
              f"{e['mean']:>+6.3f} [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]  {f['mean']:>+6.3f} [{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}]  "
              f"{r['organic_drift_final']:>+14.3f}")
    print("  start = mean starting stance (the group's natural lean); organic drift = how far the no-attacker group moves toward the target by itself.")

    out = {"kind": args.kind, "topics": rows, "pooled": {}}
    print("\nPOOLED ACROSS TOPICS")
    print(f"  {'':<14}{'random-effects (95% CI)':<34}{'fixed-effect':>14}{'tau':>8}{'I^2':>7}{'topics > 0':>12}")
    for label in ("end", "final"):
        res = pool([(r[label]["mean"], r[label]["se"]) for r in rows.values()])
        out["pooled"][label] = res
        npos = "%d/%d" % (res["n_pos"], res["k"])
        print(f"  {label:<14}{res['re']:>+7.3f} [{res['re_lo']:+.3f}, {res['re_hi']:+.3f}]      {res['fe']:>+14.3f}{res['tau']:>8.3f}"
              f"{res['i2']:>7.0%}{npos:>12}")
    dec = pool([(r["decay"]["mean"], r["decay"]["se"]) for r in rows.values()])
    out["pooled"]["decay"] = dec
    print(f"  {'decay':<14}{dec['re']:>+7.3f} [{dec['re_lo']:+.3f}, {dec['re_hi']:+.3f}]   (negative = the effect faded after the attackers went silent)")

    print("\nREADING")
    print("  - Consistent sign across topics plus a pooled interval clear of zero is the generalisation claim.")
    print("  - A large I^2 or tau means the effect depends on the topic; the per-topic intervals then matter more than the pooled one.")
    print(f"  - Only {len(rows)} topics: the pooled interval is optimistic. Report the sign count and each topic's interval too.")
    print("  - Per-topic n is 10 seeds, so single-topic p-values are descriptive.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"\n  written to {args.out}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(rows) + 2.2), sharey=True)
            names = list(rows) + ["POOLED (random effects)"]
            for ax, label in zip(axes, ("end", "final")):
                ys = list(range(len(names)))[::-1]
                for y, t in zip(ys, rows):
                    r = rows[t][label]
                    ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", color="black", capsize=3)
                res = out["pooled"][label]
                ax.errorbar(res["re"], ys[-1], xerr=[[res["re"] - res["re_lo"]], [res["re_hi"] - res["re"]]], fmt="D", color="C3", capsize=3)
                ax.axvline(0, color="grey", lw=0.8)
                ax.set_yticks(ys); ax.set_yticklabels(names)
                ax.set_title("End of attack" if label == "end" else "Final (attackers silent)")
                ax.set_xlabel("effect vs organic (toward target)")
            fig.tight_layout()
            os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
            fig.savefig(args.fig, dpi=160)
            print(f"  figure: {args.fig}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()