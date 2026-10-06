"""
Does the effect survive when attackers write messages the same length as everyone else?

Compares, on the same seeds, the same attackers and the same organic controls:
  LONG   = the direction-enforced 4-attacker arm whose messages were ~2x the length of ordinary ones
  SHORT  = the length-matched 4-attacker arm (same 1-2 short sentences instruction as ordinary agents)

Outputs, for the measured agents (never attackers in any arm):
  1. SHORT vs organic       -- is there an effect for a stealthier attacker?
  2. LONG  vs organic       -- the earlier result, on the same seeds
  3. SHORT minus LONG       -- how much of the effect depended on length? (organic cancels out of this)
  4. persistence for SHORT  -- decay after the attackers go silent
  5. per-seed table

Reading guide, fixed before the run: the interesting question is the SHORT effect and its CI. If SHORT is clearly
positive, influence does not depend on long, elaborate messages. If SHORT is near zero while LONG is positive,
the earlier effect was largely a length effect and the stealth claim fails. Intermediate results are reported
as intermediate.

Usage:
    python analyze_length_matched.py
    python analyze_length_matched.py --seeds 1 2 3 4 5 6 7 8 9 10
"""
from __future__ import annotations

import argparse
import json
import os

import analyze_confirmatory as AC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-manifest", default="runs/confirmatory_enf_manifest.json")
    ap.add_argument("--short-manifest", default="runs/confirmatory_short_manifest.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default="runs/length_matched_analysis.json")
    args = ap.parse_args()

    want = set(args.seeds) if args.seeds else None
    long_c = AC.load_cells(args.long_manifest, want)
    short_c = AC.load_cells(args.short_manifest, want)
    seeds = sorted(s for s in short_c if "attack4" in short_c[s] and s in long_c
                   and "organic" in long_c[s] and "attack4" in long_c[s])
    if len(seeds) < 3:
        print(f"Only {len(seeds)} seeds have organic + long attack4 + short attack4 -- need at least 3.")
        return

    meta0 = long_c[seeds[0]]["organic"]["meta"]
    sign = 1 if meta0["target_stance"] >= 0 else -1
    org_e, org_f, lng_e, lng_f, sht_e, sht_f = ({} for _ in range(6))
    same_attackers = True
    for s in seeds:
        measured = long_c[s]["organic"]["meta"]["measured_ids"]
        same_attackers &= long_c[s]["attack4"]["meta"]["attackers"] == short_c[s]["attack4"]["meta"]["attackers"]
        org_e[s] = AC.shift(long_c[s]["organic"]["trial"], measured, "post_stances", sign)
        org_f[s] = AC.shift(long_c[s]["organic"]["trial"], measured, "post_removal_stances", sign)
        lng_e[s] = AC.shift(long_c[s]["attack4"]["trial"], measured, "post_stances", sign)
        lng_f[s] = AC.shift(long_c[s]["attack4"]["trial"], measured, "post_removal_stances", sign)
        sht_e[s] = AC.shift(short_c[s]["attack4"]["trial"], measured, "post_stances", sign)
        sht_f[s] = AC.shift(short_c[s]["attack4"]["trial"], measured, "post_removal_stances", sign)

    out = {"seeds": seeds}
    print("=" * 100)
    print(f"LENGTH-MATCHED vs LONG-MESSAGE ATTACKERS -- 4 attackers of 8, {len(seeds)} seeds, target {meta0['target_stance']:+.2f}")
    print("=" * 100)
    print(f"  seeds: {seeds}")
    print(f"  same attackers in both arms: {'yes' if same_attackers else 'NO -- check the manifests'}")

    def block(title, rows):
        print(f"\n{title}")
        print(AC.HEADER)
        for label, d in rows:
            s_ = AC.paired_summary(d)
            out.setdefault(title, {})[label] = s_
            print(AC.row(label, s_))

    block("1-2. EFFECT vs ORGANIC", [
        ("SHORT, end of attack", [sht_e[s] - org_e[s] for s in seeds]),
        ("SHORT, final", [sht_f[s] - org_f[s] for s in seeds]),
        ("LONG,  end of attack", [lng_e[s] - org_e[s] for s in seeds]),
        ("LONG,  final", [lng_f[s] - org_f[s] for s in seeds]),
    ])
    block("3. SHORT minus LONG (negative = the effect shrank when messages were length-matched)", [
        ("end of attack", [sht_e[s] - lng_e[s] for s in seeds]),
        ("final", [sht_f[s] - lng_f[s] for s in seeds]),
    ])
    block("4. PERSISTENCE of the SHORT arm: decay = final effect minus end-of-attack effect", [
        ("SHORT decay", [(sht_f[s] - org_f[s]) - (sht_e[s] - org_e[s]) for s in seeds]),
        ("LONG decay (for comparison)", [(lng_f[s] - org_f[s]) - (lng_e[s] - org_e[s]) for s in seeds]),
    ])

    print("\n5. PER SEED (end of attack -> final; toward target; organic arm subtracted is NOT applied here)")
    print(f"  {'seed':>4}  {'organic':>16}  {'LONG':>16}  {'SHORT':>16}")
    for s in seeds:
        f = lambda e, g: f"{e[s]:+.2f} -> {g[s]:+.2f}"
        print(f"  {s:>4}  {f(org_e, org_f):>16}  {f(lng_e, lng_f):>16}  {f(sht_e, sht_f):>16}")

    # enforcement + length flags for the short arm
    ivs = [iv for s in seeds for log in short_c[s]["attack4"]["meta"]["attacker_logs"].values() for iv in log["interventions"]]
    if ivs:
        nf = sum(1 for iv in ivs if iv.get("forced_length_failed"))
        nd = sum(1 for iv in ivs if iv.get("forced_direction_failed"))
        att = [iv.get("direction_attempts", 0) for iv in ivs]
        print(f"\n6. SHORT arm generation: {len(ivs)} attacker messages; mean judge attempts {sum(att) / len(att):.2f}; "
              f"forced direction failures {nd} ({100 * nd / len(ivs):.0f}%); forced length failures {nf} ({100 * nf / len(ivs):.0f}%)")
        print("   Run `python length_check.py --manifest runs/confirmatory_short_manifest.json` to confirm the lengths now match")
        print("   (length AUC near 0.5). If they still do not, the SHORT arm is not a stealth test.")

    print("\nNOTES")
    print("  - Positive = toward the target. LONG is the earlier direction-enforced 4-attacker arm on the same seeds.")
    print("  - p-values are descriptive. The SHORT-vs-LONG contrast is within-seed, so organic noise cancels.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()