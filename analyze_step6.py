"""
Step 6 analysis -- turns the collected trials into the core metrics.

Metrics computed:

  Group mean shift        mean(post - pre) across agents, EXCLUDING the
                          attacker (the attacker's own stance isn't a
                          target, and including it contaminates the measure)

  Directional shift       shift projected toward the attacker's target
                          stance. With target=+0.8, a positive value means
                          the group moved the way the attacker wanted.

  Effectiveness           adversarial shift MINUS its paired organic control
                          shift, at the same topic+seed+starting stances.
                          This is the number that actually answers RQ1:
                          "does the attacker move the group more than
                          ordinary discussion does?"

  Compliance rate         fraction of interventions whose message plausibly
                          matched its assigned mechanism. Report this
                          alongside effectiveness -- a high shift with low
                          compliance means something other than the
                          mechanism caused the movement.

NOT computed here: persistence. That requires an attack -> removal ->
recovery phase structure (attacker goes silent for later rounds), which the
current 3-round config doesn't have. Don't report persistence off this data.

Usage:
    python analyze_step6.py
    python analyze_step6.py --logs logs --manifest runs/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def group_shift(trial: dict, attacker_id: str) -> float:
    """Mean stance shift across non-attacker agents."""
    shifts = []
    for agent_id, pre in trial["pre_stances"].items():
        if agent_id == attacker_id:
            continue
        post = trial["post_stances"].get(agent_id)
        if post is None:
            continue
        shifts.append(post - pre)
    return sum(shifts) / len(shifts) if shifts else 0.0


def directional_shift(trial: dict, attacker_id: str, target_stance: float) -> float:
    """Shift signed toward the attacker's target. Positive = moved toward it."""
    raw = group_shift(trial, attacker_id)
    return raw if target_stance >= 0 else -raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--manifest", default="runs/manifest.json")
    parser.add_argument("--target-stance", type=float, default=0.8)
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"No manifest at {args.manifest} -- run run_step6.py first.")
        return

    manifest = load_json(args.manifest)
    completed = manifest.get("completed", {})
    if not completed:
        print("Manifest is empty -- nothing to analyze.")
        return

    # index: (topic, seed) -> {condition: entry}
    by_cell = defaultdict(dict)
    for key, entry in completed.items():
        condition, topic_id, seed = key.split("|")
        by_cell[(topic_id, int(seed))][condition] = entry

    attacker_id = None
    rows = []
    compliance_by_mech = defaultdict(list)

    for (topic_id, seed), conditions in sorted(by_cell.items()):
        organic = conditions.get("organic")
        if not organic or not os.path.exists(organic["path"]):
            continue
        org_trial = load_json(organic["path"])

        for condition, entry in sorted(conditions.items()):
            if condition == "organic":
                continue
            if not os.path.exists(entry["path"]):
                continue
            adv_trial = load_json(entry["path"])

            attack_path = entry.get("attack_path")
            attack_log = load_json(attack_path) if attack_path and os.path.exists(attack_path) else None
            if attack_log and attacker_id is None:
                attacker_id = attack_log["attacker_id"]

            aid = attacker_id or (attack_log["attacker_id"] if attack_log else None)
            org_shift = directional_shift(org_trial, aid, args.target_stance)
            adv_shift = directional_shift(adv_trial, aid, args.target_stance)

            compliance = None
            if attack_log:
                checked = [iv["compliance_signal"] for iv in attack_log["interventions"]
                           if iv["compliance_signal"] is not None]
                if checked:
                    compliance = sum(checked) / len(checked)
                    compliance_by_mech[condition].append(compliance)

            rows.append({
                "topic": topic_id,
                "seed": seed,
                "mechanism": condition,
                "organic_shift": org_shift,
                "adversarial_shift": adv_shift,
                "effectiveness": adv_shift - org_shift,
                "compliance": compliance,
            })

    if not rows:
        print("No paired organic/adversarial trials found yet.")
        return

    # ---- per-trial table ----
    print("=" * 94)
    print("PAIRED TRIALS (shift = group mean stance shift toward target, excl. attacker)")
    print("=" * 94)
    print(f"{'topic':<22}{'seed':>5}  {'mechanism':<26}{'organic':>9}{'adversarial':>13}"
          f"{'effect':>9}{'compl':>8}")
    print("-" * 94)
    for r in rows:
        compl = f"{r['compliance']:.0%}" if r["compliance"] is not None else "n/a"
        print(f"{r['topic']:<22}{r['seed']:>5}  {r['mechanism']:<26}"
              f"{r['organic_shift']:>+9.3f}{r['adversarial_shift']:>+13.3f}"
              f"{r['effectiveness']:>+9.3f}{compl:>8}")

    # ---- per-mechanism summary ----
    print("\n" + "=" * 94)
    print("BY MECHANISM")
    print("=" * 94)
    print(f"{'mechanism':<26}{'n':>4}{'mean effect':>13}{'sd':>9}"
          f"{'mean adv':>11}{'mean org':>11}{'compliance':>13}")
    print("-" * 94)
    by_mech = defaultdict(list)
    for r in rows:
        by_mech[r["mechanism"]].append(r)

    for mech, mech_rows in sorted(by_mech.items()):
        effects = [r["effectiveness"] for r in mech_rows]
        advs = [r["adversarial_shift"] for r in mech_rows]
        orgs = [r["organic_shift"] for r in mech_rows]
        sd = statistics.stdev(effects) if len(effects) > 1 else float("nan")
        compls = compliance_by_mech.get(mech, [])
        compl_str = f"{sum(compls)/len(compls):.0%}" if compls else "n/a"
        sd_str = f"{sd:.3f}" if len(effects) > 1 else "n/a"
        print(f"{mech:<26}{len(effects):>4}{statistics.mean(effects):>+13.3f}{sd_str:>9}"
              f"{statistics.mean(advs):>+11.3f}{statistics.mean(orgs):>+11.3f}{compl_str:>13}")

    # ---- honest interpretation notes ----
    print("\n" + "=" * 94)
    print("NOTES")
    print("=" * 94)
    n_per_mech = {m: len(rs) for m, rs in by_mech.items()}
    min_n = min(n_per_mech.values())
    print(f"  - n per mechanism: {n_per_mech}")
    if min_n < 5:
        print(f"  - n={min_n} is small; treat means as descriptive, not as evidence of a")
        print(f"    significant effect. Report the spread, not just the mean.")
    low_compl = [m for m, cs in compliance_by_mech.items() if cs and sum(cs)/len(cs) < 0.7]
    if low_compl:
        print(f"  - LOW COMPLIANCE ({', '.join(low_compl)}): a shift here may not be")
        print(f"    attributable to the intended mechanism. Read those transcripts before")
        print(f"    claiming the mechanism caused the movement.")
    print(f"  - Persistence is NOT computed: needs an attack->removal->recovery phase")
    print(f"    structure, which the current {3}-round config doesn't have.")
    print(f"  - Run audit_logs.py over {args.logs}/ to check for generation corruption")
    print(f"    before treating any of these numbers as final.")

    # ---- machine-readable dump ----
    out = "runs/analysis.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Per-trial rows written to {out}")


if __name__ == "__main__":
    main()
