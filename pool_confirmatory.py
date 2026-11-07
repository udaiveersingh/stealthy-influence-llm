"""
Pooled analysis across models and topics, from the seed-level data (not from per-topic summaries).

  python pool_confirmatory.py                       # default cell list below; missing manifests are listed, not guessed
  python pool_confirmatory.py --cells cells.json    # [{"model": .., "topic": .., "manifest": .., "backend": ..}, ...]

Unit of analysis. One CELL = one model x one topic, n = 20 paired seeds (10 for the 32-agent runs). Within a cell the
outcome per seed is the attack arm's shift toward the target minus the organic twin's shift (same seed, same frozen
pre-stances, same measured agents), exactly as analyze_confirmatory.py computes it. The cell estimate is the mean of
those paired differences, with standard error sd / sqrt(n).

Pooling. Random-effects meta-analysis over cells (DerSimonian-Laird tau^2), with the Hartung-Knapp-Sidik-Jonkman
confidence interval (t with k-1 df), which keeps its coverage when there are only 8 cells per model. Reported for each
model (k = 8 topics) and for all three models together (k = 24), at the end of the attack and after the silent rounds:
pooled effect, 95% CI, 95% prediction interval (the range a NEW topic/model cell would be expected to fall in), tau,
I^2 and Cochran's Q. A subgroup test says whether the three models differ. Retention = pooled final / pooled end, with a
cell-level bootstrap CI. Topic consistency of persistence = Spearman correlation of the per-topic final effects between
each pair of models.

Backends. Each cell uses one backend throughout (organic and attack from the same server), so effects are always
within-backend; the backend of every cell is printed. Nemotron college_value and remote_work were only run on the
NVIDIA API; every other cell is self-hosted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics as st

from scipy import stats

import analyze_confirmatory as AC

TOPICS = ["ai_regulation", "college_value", "remote_work", "free_speech_harm", "individual_vs_collective",
          "traditional_family_roles", "centralized_governance", "religion_public_life"]
DEFAULT_TOPIC = "ai_regulation"


def manifest_for(ns, topic):
    base = f"runs/{ns}/confirmatory_short_manifest.json" if ns else "runs/confirmatory_short_manifest.json"
    return base if topic == DEFAULT_TOPIC else base.replace(".json", f"__{topic}.json")


def default_cells():
    cells = []
    for t in TOPICS:
        if t in ("college_value", "remote_work"):
            cells.append({"model": "Nemotron", "topic": t, "manifest": manifest_for("", t), "backend": "NVIDIA API"})
        else:
            cells.append({"model": "Nemotron", "topic": t, "manifest": manifest_for("hpc", t), "backend": "HPC"})
    for model, ns in (("gpt-oss", "gptoss_hpc"), ("Gemma", "gemma_hpc")):
        cells += [{"model": model, "topic": t, "manifest": manifest_for(ns, t), "backend": "HPC"} for t in TOPICS]
    return cells


SCALE_CELLS = [{"model": "Nemotron-32", "topic": t, "manifest": manifest_for("scale32_hpc", t), "backend": "HPC"}
               for t in TOPICS]


# ----------------------------------------------------------------------------------------------------------------
# cell level
# ----------------------------------------------------------------------------------------------------------------

def cell_effects(manifest, arm):
    """Per-seed paired effects (attack arm minus organic), end and final; None if the manifest is unusable."""
    if not os.path.exists(manifest):
        return None
    cells = AC.load_cells(manifest)
    seeds = sorted(s for s, c in cells.items() if "organic" in c and arm in c)
    if len(seeds) < 3:
        return None
    meta = cells[seeds[0]]["organic"]["meta"]
    sign = 1 if meta["target_stance"] >= 0 else -1
    end, fin = [], []
    for s in seeds:
        ids = cells[s]["organic"]["meta"]["measured_ids"]
        o, a = cells[s]["organic"]["trial"], cells[s][arm]["trial"]
        end.append(AC.shift(a, ids, "post_stances", sign) - AC.shift(o, ids, "post_stances", sign))
        fin.append(AC.shift(a, ids, "post_removal_stances", sign) - AC.shift(o, ids, "post_removal_stances", sign))
    keep = [i for i in range(len(seeds)) if end[i] == end[i] and fin[i] == fin[i]]
    return {"seeds": [seeds[i] for i in keep], "end": [end[i] for i in keep], "final": [fin[i] for i in keep],
            "models_in_manifest": sorted({c[a]["meta"].get("model", "?") for c in cells.values() for a in c})}


def summarise(d):
    n = len(d)
    m = st.fmean(d)
    se = st.stdev(d) / math.sqrt(n)
    return {"n": n, "mean": m, "se": se, "p": stats.ttest_1samp(d, 0).pvalue}


# ----------------------------------------------------------------------------------------------------------------
# random-effects pooling
# ----------------------------------------------------------------------------------------------------------------

def pool(ys, ses):
    k = len(ys)
    w = [1 / s ** 2 for s in ses]
    fe = sum(wi * y for wi, y in zip(w, ys)) / sum(w)
    q = sum(wi * (y - fe) ** 2 for wi, y in zip(w, ys))
    c = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = [1 / (s ** 2 + tau2) for s in ses]
    mu = sum(wi * y for wi, y in zip(ws, ys)) / sum(ws)
    # Hartung-Knapp-Sidik-Jonkman variance and t(k-1) interval
    hk = sum(wi * (y - mu) ** 2 for wi, y in zip(ws, ys)) / ((k - 1) * sum(ws))
    se_hk = math.sqrt(max(hk, 1e-12))
    tcrit = stats.t.ppf(0.975, k - 1)
    p = 2 * (1 - stats.t.cdf(abs(mu / se_hk), k - 1))
    pi_half = stats.t.ppf(0.975, max(k - 2, 1)) * math.sqrt(tau2 + se_hk ** 2)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"k": k, "mu": mu, "lo": mu - tcrit * se_hk, "hi": mu + tcrit * se_hk, "p": p, "tau": math.sqrt(tau2),
            "i2": i2, "q": q, "q_p": 1 - stats.chi2.cdf(q, k - 1), "pi_lo": mu - pi_half, "pi_hi": mu + pi_half,
            "n_pos": sum(y > 0 for y in ys)}


def boot_retention(rows, n_boot=4000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        smp = [rows[rng.randrange(len(rows))] for _ in rows]
        try:
            e = pool([r["end"]["mean"] for r in smp], [r["end"]["se"] for r in smp])["mu"]
            f = pool([r["final"]["mean"] for r in smp], [r["final"]["se"] for r in smp])["mu"]
        except ZeroDivisionError:
            continue
        if e > 0:
            vals.append(f / e)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def fmt_p(p):
    return f"{p:.3f}" if p >= 0.001 else ("<.001" if p >= 1e-6 else f"{p:.0e}")


def line(label, r):
    return (f"  {label:<22} k={r['k']:<3}{r['mu']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  p={fmt_p(r['p']):<7}"
            f" {r['n_pos']}/{r['k']} >0   PI [{r['pi_lo']:+.3f}, {r['pi_hi']:+.3f}]   tau {r['tau']:.3f}"
            f"  I2 {r['i2'] * 100:3.0f}%  Q p={fmt_p(r['q_p'])}")


def subgroup_test(groups):
    """Q-between for the model subgroups (random-effects estimates and their HK variances)."""
    mus = [g["mu"] for g in groups]
    vs = [((g["hi"] - g["lo"]) / (2 * stats.t.ppf(0.975, g["k"] - 1))) ** 2 for g in groups]
    w = [1 / v for v in vs]
    m = sum(wi * x for wi, x in zip(w, mus)) / sum(w)
    qb = sum(wi * (x - m) ** 2 for wi, x in zip(w, mus))
    return qb, 1 - stats.chi2.cdf(qb, len(groups) - 1)


# ----------------------------------------------------------------------------------------------------------------

def report(cells, arm, title, out_rows, out_json):
    rows, missing = [], []
    for c in cells:
        r = cell_effects(c["manifest"], arm)
        if r is None:
            missing.append(c)
            continue
        rows.append({**c, "arm": arm, "end": summarise(r["end"]), "final": summarise(r["final"]),
                     "served_models": r["models_in_manifest"], "seeds": r["seeds"]})
    print("=" * 112)
    print(title)
    print("=" * 112)
    if missing:
        print("  NOT FOUND (left out, not estimated): " + "; ".join(f"{m['model']}/{m['topic']} ({m['manifest']})"
                                                               for m in missing))
    print(f"\n  CELLS  (paired seeds; effect = attack minus organic twin, positive = toward the target)")
    print(f"  {'model':<12}{'topic':<27}{'backend':<12}{'n':>3}  {'end':>8} {'p':>7}   {'final':>8} {'p':>7}  retained")
    for r in rows:
        ret = f"{r['final']['mean'] / r['end']['mean'] * 100:5.0f}%" if r["end"]["p"] < 0.05 else "   n/a"
        print(f"  {r['model']:<12}{r['topic']:<27}{r['backend']:<12}{r['end']['n']:>3}  {r['end']['mean']:+8.3f}"
              f" {fmt_p(r['end']['p']):>7}   {r['final']['mean']:+8.3f} {fmt_p(r['final']['p']):>7}  {ret}")
        out_rows.append({"analysis": title.split(" --")[0], "model": r["model"], "topic": r["topic"],
                         "backend": r["backend"], "arm": arm, "n_seeds": r["end"]["n"],
                         "end_mean": round(r["end"]["mean"], 4), "end_se": round(r["end"]["se"], 4),
                         "end_p": r["end"]["p"], "final_mean": round(r["final"]["mean"], 4),
                         "final_se": round(r["final"]["se"], 4), "final_p": r["final"]["p"],
                         "served_models": "|".join(r["served_models"])})
    if not rows:
        return
    models = [m for m in dict.fromkeys(r["model"] for r in rows)]
    res = {}
    print(f"\n  RANDOM-EFFECTS POOLING (DerSimonian-Laird tau^2, Hartung-Knapp 95% CI; PI = 95% prediction interval)")
    for when in ("end", "final"):
        print(f"  -- {'END OF ATTACK (after round 3)' if when == 'end' else 'FINAL (after round 6, attackers silent)'}")
        groups = []
        for m in models:
            rs = [r for r in rows if r["model"] == m]
            if len(rs) >= 3:
                g = pool([r[when]["mean"] for r in rs], [r[when]["se"] for r in rs])
                groups.append(g); res[f"{m}|{when}"] = g
                print(line(m, g))
        if len(models) > 1:
            g = pool([r[when]["mean"] for r in rows], [r[when]["se"] for r in rows])
            res[f"ALL|{when}"] = g
            print(line("ALL", g))
            if len(groups) > 1:
                qb, qp = subgroup_test(groups)
                print(f"  {'':<22} do the models differ? Q-between {qb:.2f}, df {len(groups) - 1}, p={fmt_p(qp)}")
    print("\n  RETENTION (pooled final / pooled end; 95% CI from resampling cells)")
    for m in models + (["ALL"] if len(models) > 1 else []):
        rs = rows if m == "ALL" else [r for r in rows if r["model"] == m]
        if len(rs) >= 3 and res.get(f"{m}|end", {}).get("mu", 0) > 0:
            lo, hi = boot_retention(rs)
            print(f"  {m:<22} {res[f'{m}|final']['mu'] / res[f'{m}|end']['mu'] * 100:4.0f}%   [{lo * 100:.0f}%, {hi * 100:.0f}%]")
    if len(models) > 1:
        print("\n  IS THE SAME TOPIC PERSISTENT ON EVERY MODEL?  Spearman correlation of per-topic final effects")
        by = {m: {r["topic"]: r["final"]["mean"] for r in rows if r["model"] == m} for m in models}
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                common = sorted(set(by[a]) & set(by[b]))
                if len(common) >= 5:
                    rho = stats.spearmanr([by[a][t] for t in common], [by[b][t] for t in common])
                    print(f"  {a} vs {b}: rho {rho.statistic:+.2f} (p={fmt_p(rho.pvalue)}, {len(common)} topics)")
        print("  (near zero = which topics keep their effect is not shared across models)")
    out_json[title] = {"pooled": res, "cells": [{k: v for k, v in r.items() if k != "seeds"} for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="JSON list of cells to use instead of the default list")
    ap.add_argument("--no-scale", action="store_true", help="skip the 32-agent section")
    ap.add_argument("--out", default="results_hpc2/pooled")
    args = ap.parse_args()
    cells = json.load(open(args.cells)) if args.cells else default_cells()
    out_rows, out_json = [], {}
    report(cells, "attack4", "MAIN -- 8-agent groups, 4 of 8 covert attackers, length-matched and direction-enforced",
           out_rows, out_json)
    if not args.no_scale and not args.cells:
        for arm, lab in (("attack_high", "8 of 32"), ("attack_low", "4 of 32")):
            print()
            report(SCALE_CELLS, arm, f"SCALE -- 32-agent groups, {lab} covert attackers (Nemotron)", out_rows, out_json)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_cells.csv", "w", newline="") as f:
        if out_rows:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    json.dump(out_json, open(args.out + ".json", "w"), indent=2, default=float)
    print(f"\n  written: {args.out}_cells.csv (one row per cell, for tables/plots) and {args.out}.json")


if __name__ == "__main__":
    main()
