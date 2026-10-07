"""
Analysis of the confirmatory study (run_confirmatory.py).

Analysis choices, fixed BEFORE any result was looked at:
  * Outcome: mean stance shift of the MEASURED agents (those who are never attackers in any arm
    for that seed, so the same people are measured in every arm), signed so POSITIVE = toward the
    attacker's target. Target -0.80, so a drop in stance is positive.
  * Design: paired by seed. All arms of a seed start from identical frozen pre-stances and share
    the same speaking order, so effect = arm shift minus the organic shift of the SAME seed.
  * Two time points: end of attack (after round 3) and final (after round 6). The organic arm
    also runs six rounds, so persistence is a difference-in-differences against organic.
  * Tests: paired t and Wilcoxon signed-rank on the per-seed differences; 95% CIs from the t
    distribution. Equivalence (for a null): TOST against +/-0.10, about one run-to-run noise SD
    in the pilot. No correction across the few planned contrasts; treat p-values as descriptive.
  * Retention ratio uses a seed-level bootstrap and is flagged as uninterpretable whenever the
    end-of-attack effect's 95% CI includes zero.

Usage:
    python analyze_confirmatory.py
    python analyze_confirmatory.py --seeds 1 2 3 4 5 6 7 8 9 10      # e.g. an interim subset
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

from scipy import stats

MANIFEST_PATH = "runs/confirmatory_manifest.json"
OUT_PATH = "runs/confirmatory_analysis.json"
ARMS = ["organic", "attack1", "attack4"]
EQUIV_MARGIN = 0.10


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_cells(manifest_path, seeds=None):
    manifest = load_json(manifest_path)
    cells = defaultdict(dict)
    for key, e in manifest["completed"].items():
        if not os.path.exists(e["path"]) or not os.path.exists(e["meta_path"]):
            continue
        if seeds and e["seed"] not in seeds:
            continue
        cells[e["seed"]][e["arm"]] = {"trial": load_json(e["path"]), "meta": load_json(e["meta_path"])}
    return cells


def shift(trial, ids, field, sign, agg="mean"):
    pre, post = trial["pre_stances"], trial.get(field) or {}
    vals = [sign * (post[a] - pre[a]) for a in ids if a in pre and a in post]
    if not vals:
        return float("nan")
    return st.median(vals) if agg == "median" else sum(vals) / len(vals)


def paired_summary(d):
    n = len(d)
    m = sum(d) / n
    sd = st.stdev(d) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    p_t = stats.ttest_1samp(d, 0.0).pvalue if n > 1 and sd > 0 else float("nan")
    try:
        p_w = stats.wilcoxon(d).pvalue
    except ValueError:
        p_w = float("nan")
    return {"n": n, "mean": m, "sd": sd, "se": se, "ci_lo": m - tcrit * se, "ci_hi": m + tcrit * se,
            "p_t": p_t, "p_w": p_w, "dz": (m / sd) if sd and sd > 0 else float("nan"),
            "n_pos": sum(1 for x in d if x > 0)}


def tost(d, margin):
    n = len(d)
    m, sd = sum(d) / n, st.stdev(d)
    se = sd / math.sqrt(n)
    p_lower = 1 - stats.t.cdf((m + margin) / se, n - 1)     # H0: mean <= -margin
    p_upper = stats.t.cdf((m - margin) / se, n - 1)         # H0: mean >= +margin
    tcrit90 = stats.t.ppf(0.95, n - 1)
    return {"p": max(p_lower, p_upper), "ci90_lo": m - tcrit90 * se, "ci90_hi": m + tcrit90 * se}


def fmt_p(p):
    return "  n/a " if p is None or (isinstance(p, float) and math.isnan(p)) else (f"{p:.3f}" if p >= 0.001 else "<.001")


def row(label, s):
    pos = "%d/%d" % (s["n_pos"], s["n"])
    return ("  %-26s%3d%+9.3f%8.3f   [%+.3f, %+.3f]%8s%8s%+7.2f%7s"
            % (label, s["n"], s["mean"], s["sd"], s["ci_lo"], s["ci_hi"], fmt_p(s["p_t"]), fmt_p(s["p_w"]), s["dz"], pos))


HEADER = f"  {'':<26}{'n':>3}{'mean':>9}{'sd':>8}   {'95% CI':<18}{'p(t)':>8}{'p(W)':>8}{'dz':>7}{'>0':>7}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--fig", default="figures/confirmatory_effects.png")
    args = ap.parse_args()

    cells = load_cells(args.manifest, set(args.seeds) if args.seeds else None)
    seeds = sorted(s for s, c in cells.items() if all(a in c for a in ARMS))
    skipped = sorted(set(cells) - set(seeds))
    if len(seeds) < 3:
        print(f"Only {len(seeds)} seeds have all three arms -- need at least 3.")
        return

    any_meta = cells[seeds[0]]["organic"]["meta"]
    target = any_meta["target_stance"]
    sign = 1 if target >= 0 else -1
    topic = any_meta["topic_id"]

    # per-seed outcomes
    end, fin, end_noA7, end_med = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    fin_noA7, fin_med = defaultdict(dict), defaultdict(dict)
    for s in seeds:
        measured = cells[s]["organic"]["meta"]["measured_ids"]
        no_a7 = [a for a in measured if not a.startswith("A7")]
        for arm in ARMS:
            t = cells[s][arm]["trial"]
            end[arm][s] = shift(t, measured, "post_stances", sign)
            fin[arm][s] = shift(t, measured, "post_removal_stances", sign)
            end_noA7[arm][s] = shift(t, no_a7, "post_stances", sign)
            fin_noA7[arm][s] = shift(t, no_a7, "post_removal_stances", sign)
            end_med[arm][s] = shift(t, measured, "post_stances", sign, "median")
            fin_med[arm][s] = shift(t, measured, "post_removal_stances", sign, "median")

    def eff(table, arm):
        return [table[arm][s] - table["organic"][s] for s in seeds]

    out = {"seeds": seeds, "target": target, "topic": topic}
    print("=" * 100)
    print(f"CONFIRMATORY ANALYSIS -- {topic}, target {target:+.2f}  (POSITIVE = group moved toward the target)")
    print("=" * 100)
    n_meas = len(cells[seeds[0]]["organic"]["meta"]["measured_ids"])
    print(f"  seeds analysed: {len(seeds)} {seeds}" + (f"   (incomplete, skipped: {skipped})" if skipped else ""))
    print(f"  measured agents per seed: {n_meas} (never attackers in any arm; same people in every arm)")

    # 1. organic baseline drift
    print("\n1. ORGANIC BASELINE DRIFT (no attackers): how far groups move toward the target on their own")
    print(HEADER)
    ob_end = paired_summary([end["organic"][s] for s in seeds])
    ob_fin = paired_summary([fin["organic"][s] for s in seeds])
    print(row("after round 3", ob_end)); print(row("after round 6", ob_fin))
    out["organic_drift"] = {"end": ob_end, "final": ob_fin}

    # 2/3. effects vs organic
    out["effect_end"], out["effect_final"] = {}, {}
    for title, table, key in (("2. EFFECT vs ORGANIC, END OF ATTACK (after round 3)", end, "effect_end"),
                              ("3. EFFECT vs ORGANIC, FINAL (after round 6, attackers silent since round 3)", fin, "effect_final")):
        print(f"\n{title}")
        print(HEADER)
        for arm in ("attack1", "attack4"):
            s_ = paired_summary(eff(table, arm)); out[key][arm] = s_
            print(row(arm + f" ({1 if arm == 'attack1' else 4} of 8)", s_))

    # 4. dose-response
    print("\n4. DOSE-RESPONSE: attack4 minus attack1 (does adding attackers add effect?)")
    print(HEADER)
    out["dose"] = {}
    for label, table in (("end of attack", end), ("final", fin)):
        s_ = paired_summary([table["attack4"][s] - table["attack1"][s] for s in seeds])
        out["dose"][label] = s_
        print(row(label, s_))

    # 5. persistence
    print("\n5. PERSISTENCE: does the effect survive three silent rounds?")
    print("  decay = final effect minus end-of-attack effect, each against organic (negative = faded)")
    print(HEADER)
    out["persistence"] = {}
    rng = random.Random(0)
    for arm in ("attack1", "attack4"):
        e_end, e_fin = eff(end, arm), eff(fin, arm)
        decay = paired_summary([f - e for e, f in zip(e_end, e_fin)])
        print(row(f"{arm} decay", decay))
        m_end = sum(e_end) / len(e_end); m_fin = sum(e_fin) / len(e_fin)
        boots = []
        for _ in range(5000):
            idx = [rng.randrange(len(seeds)) for _ in seeds]
            me = sum(e_end[i] for i in idx) / len(idx); mf = sum(e_fin[i] for i in idx) / len(idx)
            if abs(me) > 1e-9:
                boots.append(mf / me)
        boots.sort()
        lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (float("nan"),) * 2
        end_ci = out["effect_end"][arm]
        ok = end_ci["ci_lo"] > 0 or end_ci["ci_hi"] < 0
        ratio = m_fin / m_end if abs(m_end) > 1e-9 else float("nan")
        if ok:
            print(f"    {arm}: retained {ratio:.2f}  (bootstrap 95% CI {lo:.2f} to {hi:.2f})")
        else:
            print(f"    {arm}: retention ratio not reported -- the end-of-attack effect is not distinguishable "
                  f"from zero, so there is nothing to retain")
        out["persistence"][arm] = {"decay": decay, "ratio": ratio, "ratio_ci": [lo, hi], "interpretable": ok}

    # 6. equivalence for the single attacker
    print(f"\n6. EQUIVALENCE: is the 1-attacker effect within +/-{EQUIV_MARGIN:.2f} of zero? (TOST)")
    out["equivalence"] = {}
    for label, table in (("end of attack", end), ("final", fin)):
        d = eff(table, "attack1")
        t_ = tost(d, EQUIV_MARGIN)
        verdict = "EQUIVALENT (effect is smaller than the margin)" if t_["p"] < 0.05 else "not shown to be equivalent"
        print(f"  attack1, {label:<14} 90% CI [{t_['ci90_lo']:+.3f}, {t_['ci90_hi']:+.3f}]   TOST p = {fmt_p(t_['p'])}   {verdict}")
        out["equivalence"][label] = t_
    print(f"  (margin +/-{EQUIV_MARGIN:.2f} = about one run-to-run noise SD seen in the pilot; a judgment call, stated up front)")

    # 7. robustness
    print("\n7. ROBUSTNESS of the end-of-attack effect (attack4 / attack1)")
    out["robustness"] = {}
    for label, table in (("main: mean of 4 measured", end), ("excluding A7_STATUS (3 agents)", end_noA7),
                         ("median of measured agents", end_med)):
        a4 = paired_summary(eff(table, "attack4")); a1 = paired_summary(eff(table, "attack1"))
        out["robustness"][label] = {"attack4": a4, "attack1": a1}
        print(f"  {label:<34} attack4 {a4['mean']:+.3f} [{a4['ci_lo']:+.3f},{a4['ci_hi']:+.3f}]   "
              f"attack1 {a1['mean']:+.3f} [{a1['ci_lo']:+.3f},{a1['ci_hi']:+.3f}]")

    # 8. per-seed table
    print("\n8. PER SEED (shift toward target of the measured agents)")
    print(f"  {'seed':>4}  {'organic':>16}  {'attack1':>16}  {'attack4':>16}   attackers(attack4)")
    for s in seeds:
        f = lambda arm: f"{end[arm][s]:+.2f} -> {fin[arm][s]:+.2f}"
        print(f"  {s:>4}  {f('organic'):>16}  {f('attack1'):>16}  {f('attack4'):>16}   "
              f"{','.join(a.split('_')[0] for a in cells[s]['attack4']['meta']['attackers'])}")
    print("  (each cell: end of attack -> final)")

    print("\n" + "=" * 100)
    print("NOTES")
    print("=" * 100)
    print("  - Positive = toward the target. Organic groups drift toward it on their own (section 1), which")
    print("    is why every effect is measured against the same seed's organic arm.")
    print("  - The manipulation check (did attackers actually argue toward the target?) is a separate step.")
    print("    Until it is done, do not describe a null as 'groups resisted'.")
    print("  - p-values are descriptive: few planned contrasts, small n, and no multiplicity correction.")
    if args.seeds is None and len(seeds) < 20:
        print(f"  - n = {len(seeds)} seeds. Treat as interim if more seeds are planned.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  written to {args.out}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
            for ax, (title, table) in zip(axes, (("End of attack (round 3)", end), ("Final (round 6)", fin))):
                for j, arm in enumerate(("attack1", "attack4")):
                    d = eff(table, arm)
                    xs = [j + (random.Random(i).random() - 0.5) * 0.25 for i in range(len(d))]
                    ax.scatter(xs, d, alpha=0.55, s=22)
                    s_ = paired_summary(d)
                    ax.errorbar([j], [s_["mean"]], yerr=[[s_["mean"] - s_["ci_lo"]], [s_["ci_hi"] - s_["mean"]]],
                                fmt="D", color="black", capsize=4)
                ax.axhline(0, color="grey", lw=0.8)
                ax.set_xticks([0, 1]); ax.set_xticklabels(["1 attacker", "4 attackers"])
                ax.set_title(title)
            axes[0].set_ylabel("effect vs organic (toward target = positive)")
            fig.tight_layout()
            os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
            fig.savefig(args.fig, dpi=160)
            print(f"  figure: {args.fig}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()