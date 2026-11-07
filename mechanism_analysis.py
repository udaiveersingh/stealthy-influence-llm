"""
EXPLORATORY: which cell properties go with an effect that lasts, fades or backfires? (no API calls)

Over the 24 model x topic cells of the main study, cell-level properties measured on the organic twin and the frozen
pre-stances (so none of them depends on the attack) are related to three outcomes of the attack:
    end   = effect at the end of the attack,  final = effect after the silent rounds,  decay = final - end
Properties (all signed so POSITIVE = toward the attackers' target side):
    organic_drift_end / organic_drift_final   how far the group moves toward the target with no attackers
    pre_lean      mean pre-stance of the measured agents, toward the target (how far they already lean that way)
    pre_spread    SD of the measured agents' pre-stances (how much they disagree at the start)
    headroom      mean distance from the pre-stances to the target stance (room left to move)
Each outcome is regressed on one property at a time with a random-effects meta-regression (weights 1/(se^2 + tau^2),
tau^2 from DerSimonian-Laird; t test with k-2 df on the Knapp-Hartung-scaled SE) and a Spearman correlation is shown
alongside. Five properties x three outcomes = 15 tests on 24 cells: this is hypothesis-generating, so nothing here is
claimed as confirmatory, and the listed p-values are uncorrected.

  python mechanism_analysis.py
"""
from __future__ import annotations

import math
import statistics as st

import numpy as np
from scipy import stats

import analyze_confirmatory as AC
import pool_confirmatory as P


def covariates(manifest):
    cells = AC.load_cells(manifest)
    seeds = sorted(s for s, c in cells.items() if "organic" in c and "attack4" in c)
    meta = cells[seeds[0]]["organic"]["meta"]
    tgt = meta["target_stance"]
    sign = 1 if tgt >= 0 else -1
    d_end, d_fin, lean, spread, room = [], [], [], [], []
    for s in seeds:
        o = cells[s]["organic"]
        ids = o["meta"]["measured_ids"]
        pre = [o["trial"]["pre_stances"][a] for a in ids if a in o["trial"]["pre_stances"]]
        d_end.append(AC.shift(o["trial"], ids, "post_stances", sign))
        d_fin.append(AC.shift(o["trial"], ids, "post_removal_stances", sign))
        lean.append(st.fmean(sign * p for p in pre))
        spread.append(st.pstdev(pre))
        room.append(st.fmean(abs(tgt - p) for p in pre))
    f = lambda v: st.fmean(x for x in v if x == x)
    return {"organic_drift_end": f(d_end), "organic_drift_final": f(d_fin), "pre_lean": f(lean),
            "pre_spread": f(spread), "headroom": f(room)}


def metareg(y, se, x):
    y, se, x = map(np.asarray, (y, se, x))
    k = len(y)
    tau2 = P.pool(list(y), list(se))["tau"] ** 2
    w = 1 / (se ** 2 + tau2)
    X = np.column_stack([np.ones(k), x])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    beta = np.linalg.solve(XtWX, X.T @ W @ y)
    resid = y - X @ beta
    s2 = float(resid @ W @ resid) / (k - 2)                      # Knapp-Hartung scale
    cov = np.linalg.inv(XtWX) * max(s2, 1e-12)
    b, sb = beta[1], math.sqrt(cov[1, 1])
    p = 2 * (1 - stats.t.cdf(abs(b / sb), k - 2))
    return b, sb, p


def main():
    rows = []
    for c in P.default_cells():
        eff = P.cell_effects(c["manifest"], "attack4")
        if eff is None:
            continue
        cov = covariates(c["manifest"])
        e, f = P.summarise(eff["end"]), P.summarise(eff["final"])
        dec = [b - a for a, b in zip(eff["end"], eff["final"])]
        rows.append({**c, **cov, "end": e, "final": f, "decay": P.summarise(dec)})
    print("=" * 108)
    print(f"EXPLORATORY MECHANISM ANALYSIS -- {len(rows)} cells (attack4 vs organic twin); properties from the organic "
          f"twin and pre-stances")
    print("=" * 108)
    print(f"\n  {'model':<10}{'topic':<27}{'end':>7}{'final':>8}{'decay':>8}   {'drift_end':>9}{'drift_fin':>10}"
          f"{'pre_lean':>9}{'spread':>8}{'headroom':>9}")
    for r in sorted(rows, key=lambda r: r["final"]["mean"]):
        print(f"  {r['model']:<10}{r['topic']:<27}{r['end']['mean']:+7.3f}{r['final']['mean']:+8.3f}"
              f"{r['decay']['mean']:+8.3f}   {r['organic_drift_end']:+9.3f}{r['organic_drift_final']:+10.3f}"
              f"{r['pre_lean']:+9.3f}{r['pre_spread']:8.3f}{r['headroom']:9.3f}")
    props = ["organic_drift_end", "organic_drift_final", "pre_lean", "pre_spread", "headroom"]
    print("\n  META-REGRESSION, one property at a time  (slope per unit of the property; KH t-test, k-2 df; Spearman rho)")
    print(f"  {'outcome':<8}{'property':<22}{'slope':>9}{'SE':>8}{'p':>8}   {'rho':>6}{'p(rho)':>8}")
    for out in ("end", "final", "decay"):
        for pr in props:
            y = [r[out]["mean"] for r in rows]
            se = [r[out]["se"] for r in rows]
            x = [r[pr] for r in rows]
            b, sb, p = metareg(y, se, x)
            rho = stats.spearmanr(x, y)
            print(f"  {out:<8}{pr:<22}{b:+9.3f}{sb:8.3f}{p:8.3f}   {rho.statistic:+6.2f}{rho.pvalue:8.3f}")
        print()
    print("  Exploratory: 15 uncorrected tests on 24 cells; a Bonferroni threshold would be p < .0033. Report any pattern "
          "as a\n  hypothesis for future work, and check it is not carried by one model (rerun after dropping each model).")
    for drop in dict.fromkeys(r["model"] for r in rows):
        sub = [r for r in rows if r["model"] != drop]
        b, sb, p = metareg([r["final"]["mean"] for r in sub], [r["final"]["se"] for r in sub],
                           [r["organic_drift_end"] for r in sub])
        print(f"  without {drop:<9} final ~ organic_drift_end: slope {b:+.3f} (SE {sb:.3f}, p={p:.3f}, k={len(sub)})")


if __name__ == "__main__":
    main()
