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
    raw = group_shift(trial, attacker_id)
    return raw if target_stance >= 0 else -raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--manifest", default="runs/manifest.json")
    parser.add_argument("--target-stance", type=float, default=-0.8)
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"No manifest at {args.manifest} -- run run_step6.py first.")
        return

    manifest = load_json(args.manifest)
    completed = manifest.get("completed", {})
    if not completed:
        print("Manifest is empty -- nothing to analyze.")
        return

    by_cell = defaultdict(dict)
    for key, entry in completed.items():
        condition, topic_id, seed = key.split("|")
        by_cell[(topic_id, int(seed))][condition] = entry

    attacker_id = None
    rows = []
    compliance_by_mech = defaultdict(list)
    forced_failures_by_mech = defaultdict(int)
    enforced_by_mech = defaultdict(bool)

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
                interventions = attack_log["interventions"]
                checked = [iv["compliance_signal"] for iv in interventions
                           if iv["compliance_signal"] is not None]
                if checked:
                    compliance = sum(checked) / len(checked)
                    compliance_by_mech[condition].append(compliance)
                if any(iv.get("compliance_enforced") for iv in interventions):
                    enforced_by_mech[condition] = True
                forced_failures_by_mech[condition] += sum(
                    1 for iv in interventions if iv.get("forced_compliance_failed")
                )

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

    print("\n" + "=" * 94)
    print("BY MECHANISM")
    print("=" * 94)
    print(f"{'mechanism':<26}{'n':>4}{'mean effect':>13}{'sd':>9}"
          f"{'mean adv':>11}{'mean org':>11}{'compliance':>13}{'enforced':>10}{'forced-fail':>12}")
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
        enforced_str = "yes" if enforced_by_mech.get(mech) else "no"
        forced_fail = forced_failures_by_mech.get(mech, 0)
        print(f"{mech:<26}{len(effects):>4}{statistics.mean(effects):>+13.3f}{sd_str:>9}"
              f"{statistics.mean(advs):>+11.3f}{statistics.mean(orgs):>+11.3f}{compl_str:>13}"
              f"{enforced_str:>10}{forced_fail:>12}")

    print("\n" + "=" * 94)
    print("NOTES")
    print("=" * 94)
    n_per_mech = {m: len(rs) for m, rs in by_mech.items()}
    min_n = min(n_per_mech.values())
    print(f"  - n per mechanism: {n_per_mech}")
    if min_n < 13:
        print(f"  - n={min_n} is below the ~13-15 needed for 80% power at the Run-1 effect sizes")
        print(f"    (d=-0.80/-0.88). Treat means as descriptive until n increases.")
    unenforced_checkable = [m for m in by_mech if compliance_by_mech.get(m) and not enforced_by_mech.get(m)]
    if unenforced_checkable:
        print(f"  - UNENFORCED compliance ({', '.join(unenforced_checkable)}): compliance here is")
        print(f"    an OUTCOME of the model's behavior, not an assigned treatment -- any")
        print(f"    compliance-vs-effect correlation for these is exploratory, not causal.")
    enforced_mechs = [m for m, v in enforced_by_mech.items() if v]
    if enforced_mechs:
        print(f"  - ENFORCED compliance ({', '.join(enforced_mechs)}): compliance was assigned via")
        print(f"    retry-until-compliant, so a comparison between these mechanisms' effects IS")
        print(f"    a controlled test of assertion style, holding claim content constant.")
    any_forced_fail = sum(forced_failures_by_mech.values())
    if any_forced_fail:
        print(f"  - {any_forced_fail} intervention(s) had forced_compliance_failed=True -- even")
        print(f"    enforcement couldn't get a compliant generation within the retry budget.")
        print(f"    Worth reading those specific messages before trusting their trial's numbers.")
    print(f"  - Persistence is NOT computed: needs an attack->removal->recovery phase")
    print(f"    structure, which the current 3-round config doesn't have.")
    print(f"  - Run audit_logs.py over {args.logs}/ to check for generation corruption")
    print(f"    before treating any of these numbers as final.")

    out = "runs/analysis.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Per-trial rows written to {out}")


if __name__ == "__main__":
    main()