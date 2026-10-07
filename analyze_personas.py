"""
Which attacker personality shifts a group the most? (and does HIGH STATUS help?)

One attacker per trial; WHO it is varies: A3 (skeptical/contrarian), A6 (trusting/agreeable), A7 (HIGH STATUS: the
visible 'respected senior community member'), A8 (ordinary, thoughtful participant). Each trial is length-matched and
direction-enforced, so personas are compared like for like. The measured agents are fixed (A1, A2, A4, A5; never an
attacker in any persona arm) and every effect is measured against the same seed's organic arm.

Fixed BEFORE any result was seen:
  PRIMARY contrast: A7 (high status) minus A8 (ordinary), final reading, paired by seed. This is the high-status test.
  Everything else is exploratory: effect of each persona, the omnibus 'do personas differ at all?' test, and the
  pairwise contrasts (reported with Bonferroni-adjusted p for the 6 pairs).

Two different things are reported per persona, because they are not the same:
  OBEDIENCE   how readily the persona argued the assigned side (first-draft toward rate, attempts, forced failures)
  PERSUASION  how far the group moved (the effect)

Usage:
    python analyze_personas.py
    python analyze_personas.py --manifest runs/persona_manifest.json --seeds 1 2 3 4 5 6 7 8 9 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st

from scipy import stats

import analyze_confirmatory as AC

PERSONAS = {"persona_A3": "A3 contrarian", "persona_A6": "A6 trusting", "persona_A7": "A7 HIGH STATUS", "persona_A8": "A8 ordinary"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/persona_manifest.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default="runs/persona_analysis.json")
    args = ap.parse_args()

    cells = AC.load_cells(args.manifest, set(args.seeds) if args.seeds else None)
    seeds = sorted(s for s, c in cells.items() if "organic" in c and all(a in c for a in PERSONAS))
    if len(seeds) < 3:
        print(f"Only {len(seeds)} seeds have the organic arm and all four persona arms -- need at least 3.")
        return
    meta0 = cells[seeds[0]]["persona_A3"]["meta"]
    measured = meta0["measured_ids"]
    sign = 1 if meta0["target_stance"] >= 0 else -1

    end, fin = {"organic": {}}, {"organic": {}}
    for s in seeds:
        o = cells[s]["organic"]["trial"]
        end["organic"][s] = AC.shift(o, measured, "post_stances", sign)
        fin["organic"][s] = AC.shift(o, measured, "post_removal_stances", sign)
    for arm in PERSONAS:
        end[arm], fin[arm] = {}, {}
        for s in seeds:
            t = cells[s][arm]["trial"]
            end[arm][s] = AC.shift(t, measured, "post_stances", sign)
            fin[arm][s] = AC.shift(t, measured, "post_removal_stances", sign)
    eff_e = {a: [end[a][s] - end["organic"][s] for s in seeds] for a in PERSONAS}
    eff_f = {a: [fin[a][s] - fin["organic"][s] for s in seeds] for a in PERSONAS}

    out = {"seeds": seeds, "measured": measured}
    print("=" * 100)
    print(f"PERSONA SWEEP -- one attacker per trial, {len(seeds)} seeds, target {meta0['target_stance']:+.2f} "
          f"(POSITIVE = group moved toward the target)")
    print("=" * 100)
    print(f"  measured agents (fixed, same people in every arm): {', '.join(a.split('_')[0] for a in measured)}")

    print("\n1. EFFECT vs ORGANIC, by attacker persona")
    print(AC.HEADER)
    out["effects"] = {}
    for label, eff in (("end of attack", eff_e), ("final", eff_f)):
        for arm, name in PERSONAS.items():
            s_ = AC.paired_summary(eff[arm])
            out["effects"][f"{arm}|{label}"] = s_
            print(AC.row(f"{name}, {label}", s_))
        print()

    print("2. PRIMARY CONTRAST (pre-specified): does HIGH STATUS help?   A7 minus A8, same seeds")
    print(AC.HEADER)
    out["status_contrast"] = {}
    for label, eff in (("end of attack", eff_e), ("final", eff_f)):
        d = [a - b for a, b in zip(eff["persona_A7"], eff["persona_A8"])]
        s_ = AC.paired_summary(d)
        out["status_contrast"][label] = s_
        print(AC.row(f"A7 - A8, {label}", s_))
    print("   positive = the high-status attacker moved the group MORE than the ordinary attacker.")

    print("\n3. DO PERSONAS DIFFER AT ALL?  (Friedman test across the four personas, paired by seed)")
    out["friedman"] = {}
    for label, eff in (("end of attack", eff_e), ("final", eff_f)):
        try:
            chi, p = stats.friedmanchisquare(*[eff[a] for a in PERSONAS])
        except ValueError:
            chi, p = float("nan"), float("nan")
        out["friedman"][label] = {"chi2": chi, "p": p}
        print(f"   {label:<14} chi2 = {chi:.2f}, p = {AC.fmt_p(p).strip()}")
    print("   If this is not small, do not read the pairwise table below as 'persona X beats persona Y'.")

    print("\n4. PAIRWISE (final reading; Bonferroni over 6 pairs). Exploratory.")
    print(f"   {'pair':<34}{'mean diff':>10}   {'95% CI':<20}{'p':>8}{'p (adj)':>9}")
    for a, b in itertools.combinations(PERSONAS, 2):
        d = [x - y for x, y in zip(eff_f[a], eff_f[b])]
        s_ = AC.paired_summary(d)
        padj = min(1.0, s_["p_t"] * 6) if s_["p_t"] == s_["p_t"] else float("nan")
        print(f"   {PERSONAS[a] + ' - ' + PERSONAS[b]:<34}{s_['mean']:>+10.3f}   [{s_['ci_lo']:+.3f}, {s_['ci_hi']:+.3f}]"
              f"{AC.fmt_p(s_['p_t']):>8}{AC.fmt_p(padj):>9}")

    print("\n5. OBEDIENCE vs PERSUASION, per persona")
    print("   obedience = how readily the persona argued the assigned side; persuasion = how far the group moved (final effect)")
    print(f"   {'persona':<18}{'msgs':>6}{'first draft toward':>20}{'mean attempts':>15}{'forced dir.':>13}{'forced len.':>13}{'persuasion':>12}")
    out["obedience"] = {}
    for arm, name in PERSONAS.items():
        ivs = [iv for s in seeds for log in cells[s][arm]["meta"]["attacker_logs"].values() for iv in log["interventions"]]
        if not ivs:
            print(f"   {name:<18}  no intervention logs")
            continue
        att = [iv.get("direction_attempts", 0) for iv in ivs]
        first = sum(1 for x in att if x == 1) / len(ivs)
        fd = sum(1 for iv in ivs if iv.get("forced_direction_failed")) / len(ivs)
        fl = sum(1 for iv in ivs if iv.get("forced_length_failed")) / len(ivs)
        pers = st.mean(eff_f[arm])
        out["obedience"][arm] = {"n": len(ivs), "first_draft": first, "attempts": st.mean(att), "forced_direction": fd,
                                 "forced_length": fl, "persuasion_final": pers}
        print(f"   {name:<18}{len(ivs):>6}{first:>20.0%}{st.mean(att):>15.2f}{fd:>13.0%}{fl:>13.0%}{pers:>+12.3f}")
    print("   A persona that needs many attempts but then moves the group is persuasive once it complies; one that complies")
    print("   easily but moves nobody is obedient and unpersuasive. Report both.")

    print("\n6. DID THE ATTACKER'S OWN STANCE MOVE?  (attacker's shift minus the same agent's shift in the organic arm)")
    print(AC.HEADER)
    for arm, name in PERSONAS.items():
        aid = cells[seeds[0]][arm]["meta"]["attackers"][0]
        d = [AC.shift(cells[s][arm]["trial"], [aid], "post_stances", sign) - AC.shift(cells[s]["organic"]["trial"], [aid], "post_stances", sign)
             for s in seeds]
        print(AC.row(f"{name}, end of attack", AC.paired_summary(d)))

    print("\n7. PER SEED (final effect vs organic)")
    print(f"   {'seed':>4}" + "".join(f"{PERSONAS[a]:>18}" for a in PERSONAS))
    for i, s in enumerate(seeds):
        print(f"   {s:>4}" + "".join(f"{eff_f[a][i]:>+18.2f}" for a in PERSONAS))

    print("\nNOTES")
    print("  - One attacker moves a group only a little (pooled earlier: about +0.05 to +0.11), so only large persona")
    print("    differences can show with this many seeds. A null persona result is 'no large difference detected'.")
    print("  - One topic, one target direction, one model. The 'status' here is a visible label plus persona text.")
    print("  - Everything except the A7-A8 contrast is exploratory.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()