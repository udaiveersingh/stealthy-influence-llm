from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _shift_at(trial: dict, attacker_id: str, stance_field: str) -> float:
    """Mean stance shift (excl. attacker) from pre_stances to the given
    stance field -- 'post_stances' (end of attack) or 'post_removal_stances'
    (after the attacker has gone silent, only present on persistence trials)."""
    shifts = []
    target_stances = trial.get(stance_field) or {}
    for agent_id, pre in trial["pre_stances"].items():
        if agent_id == attacker_id:
            continue
        post = target_stances.get(agent_id)
        if post is None:
            continue
        shifts.append(post - pre)
    return sum(shifts) / len(shifts) if shifts else 0.0


def _signed(raw: float, target_stance: float) -> float:
    """Positive = moved toward the attacker's target, regardless of the
    target's own sign."""
    return raw if target_stance >= 0 else -raw


def parse_key(key: str):
    """Keys are condition|topic|seed for attack-only trials, or
    condition|topic|seed|removalN for persistence trials. Returns
    (condition, topic_id, seed, removal_suffix_or_None)."""
    parts = key.split("|")
    condition, topic_id, seed = parts[0], parts[1], int(parts[2])
    removal_suffix = parts[3] if len(parts) > 3 else None
    return condition, topic_id, seed, removal_suffix


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

    # by_cell[(topic, seed)][label] = entry -- label is the mechanism name for
    # attack-only trials, or "mechanism+removalN" for persistence trials, so
    # the two never collide when both exist for the same (topic, seed, mechanism).
    by_cell = defaultdict(dict)
    for key, entry in completed.items():
        condition, topic_id, seed, removal_suffix = parse_key(key)
        label = f"{condition}+{removal_suffix}" if removal_suffix else condition
        by_cell[(topic_id, seed)][label] = entry

    attacker_id = None
    rows = []
    persistence_rows = []
    compliance_by_mech = defaultdict(list)
    forced_failures_by_mech = defaultdict(int)
    enforced_by_mech = defaultdict(bool)

    for (topic_id, seed), conditions in sorted(by_cell.items()):
        organic = conditions.get("organic")
        if not organic or not os.path.exists(organic["path"]):
            continue
        org_trial = load_json(organic["path"])

        for label, entry in sorted(conditions.items()):
            if label == "organic":
                continue
            if not os.path.exists(entry["path"]):
                continue
            adv_trial = load_json(entry["path"])

            attack_path = entry.get("attack_path")
            attack_log = load_json(attack_path) if attack_path and os.path.exists(attack_path) else None
            if attack_log and attacker_id is None:
                attacker_id = attack_log["attacker_id"]
            aid = attacker_id or (attack_log["attacker_id"] if attack_log else None)

            org_shift = _signed(_shift_at(org_trial, aid, "post_stances"), args.target_stance)
            adv_shift = _signed(_shift_at(adv_trial, aid, "post_stances"), args.target_stance)

            compliance = None
            if attack_log:
                interventions = attack_log["interventions"]
                checked = [iv["compliance_signal"] for iv in interventions
                           if iv["compliance_signal"] is not None]
                if checked:
                    compliance = sum(checked) / len(checked)
                    compliance_by_mech[label].append(compliance)
                if any(iv.get("compliance_enforced") for iv in interventions):
                    enforced_by_mech[label] = True
                forced_failures_by_mech[label] += sum(
                    1 for iv in interventions if iv.get("forced_compliance_failed")
                )

            rows.append({
                "topic": topic_id, "seed": seed, "mechanism": label,
                "organic_shift": org_shift, "adversarial_shift": adv_shift,
                "effectiveness": adv_shift - org_shift, "compliance": compliance,
            })

            if adv_trial.get("post_removal_stances"):
                attack_effect_signed = adv_shift
                post_removal_signed = _signed(
                    _shift_at(adv_trial, aid, "post_removal_stances"), args.target_stance
                )
                recovery = post_removal_signed - attack_effect_signed
                # Below this, dividing produces noise amplified into a huge
                # number, not a meaningful ratio -- confirmed empirically:
                # attack_effect=0.036 and 0.043 produced ratios of +4.60 and
                # +5.33, with no values in between, which is the signature of
                # near-zero-denominator instability, not real variation.
                MIN_EFFECT_FOR_RATIO = 0.10
                ratio = (post_removal_signed / attack_effect_signed) \
                    if abs(attack_effect_signed) > MIN_EFFECT_FOR_RATIO else None
                persistence_rows.append({
                    "topic": topic_id, "seed": seed, "mechanism": label,
                    "attack_effect": attack_effect_signed,
                    "post_removal_effect": post_removal_signed,
                    "recovery": recovery,
                    "ratio": ratio,
                })

    if not rows:
        print("No paired organic/adversarial trials found yet.")
        return

    print("=" * 94)
    print("PAIRED TRIALS (shift = group mean stance shift toward target, excl. attacker)")
    print("=" * 94)
    print(f"{'topic':<22}{'seed':>5}  {'mechanism':<30}{'organic':>9}{'adversarial':>13}"
          f"{'effect':>9}{'compl':>8}")
    print("-" * 94)
    for r in rows:
        compl = f"{r['compliance']:.0%}" if r["compliance"] is not None else "n/a"
        print(f"{r['topic']:<22}{r['seed']:>5}  {r['mechanism']:<30}"
              f"{r['organic_shift']:>+9.3f}{r['adversarial_shift']:>+13.3f}"
              f"{r['effectiveness']:>+9.3f}{compl:>8}")

    print("\n" + "=" * 94)
    print("BY MECHANISM")
    print("=" * 94)
    print(f"{'mechanism':<30}{'n':>4}{'mean effect':>13}{'sd':>9}"
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
        print(f"{mech:<30}{len(effects):>4}{statistics.mean(effects):>+13.3f}{sd_str:>9}"
              f"{statistics.mean(advs):>+11.3f}{statistics.mean(orgs):>+11.3f}{compl_str:>13}"
              f"{enforced_str:>10}{forced_fail:>12}")

    if persistence_rows:
        print("\n" + "=" * 94)
        print("PERSISTENCE (does the shift survive the attacker going silent?)")
        print("=" * 94)
        print("  attack_effect      = signed shift toward target at end of attack phase")
        print("  post_removal       = signed shift toward target after removal rounds")
        print("  recovery           = post_removal - attack_effect (negative = decayed back)")
        print("                       PRIMARY metric -- an additive difference, doesn't blow up")
        print("                       when attack_effect is small, unlike ratio below")
        print("  ratio              = post_removal / attack_effect (1.0 = fully retained,")
        print("                       0 = fully decayed, negative = reversed past baseline)")
        print("                       SECONDARY -- suppressed ('n/a') below a minimum attack")
        print("                       effect, since dividing by a small number amplifies noise")
        print("                       into large, meaningless values. Read recovery first.")
        print("-" * 94)
        print(f"{'topic':<22}{'seed':>5}  {'mechanism':<28}{'attack_effect':>14}"
              f"{'post_removal':>13}{'recovery':>10}{'ratio':>8}")
        for pr in persistence_rows:
            ratio_str = f"{pr['ratio']:+.2f}" if pr["ratio"] is not None else "n/a"
            print(f"{pr['topic']:<22}{pr['seed']:>5}  {pr['mechanism']:<28}"
                  f"{pr['attack_effect']:>+14.3f}{pr['post_removal_effect']:>+13.3f}"
                  f"{pr['recovery']:>+10.3f}{ratio_str:>8}")

        print("\nBY MECHANISM (persistence)")
        print("-" * 94)
        by_mech_persist = defaultdict(list)
        for pr in persistence_rows:
            by_mech_persist[pr["mechanism"]].append(pr)
        for mech, mech_rows in sorted(by_mech_persist.items()):
            ratios = [r["ratio"] for r in mech_rows if r["ratio"] is not None]
            recoveries = [r["recovery"] for r in mech_rows]
            n_undefined = len(mech_rows) - len(ratios)
            ratio_str = f"{statistics.mean(ratios):+.2f}" if ratios else "n/a"
            note = f" ({n_undefined} n/a, attack effect too small)" if n_undefined else ""
            print(f"  {mech:<28} n={len(mech_rows)}  mean recovery={statistics.mean(recoveries):+.3f}  "
                  f"mean ratio={ratio_str}{note}")

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
    if persistence_rows:
        small_effect_n = sum(1 for pr in persistence_rows if pr["ratio"] is None)
        if small_effect_n:
            print(f"  - {small_effect_n} persistence trial(s) had an attack effect too small to")
            print(f"    divide by meaningfully -- their ratio is 'n/a' by design, not a bug.")
        print(f"  - The organic control still runs only the attack-phase round count -- it has")
        print(f"    no matching 'removal phase' of its own, so persistence numbers show whether")
        print(f"    THIS trial's shift decayed, not whether that decay differs from what organic")
        print(f"    conversation does over the same extra rounds. That comparison isn't built yet.")
    print(f"  - Run audit_logs.py over {args.logs}/ to check for generation corruption")
    print(f"    before treating any of these numbers as final.")

    out = "runs/analysis.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"effectiveness": rows, "persistence": persistence_rows}, f, indent=2)
    print(f"\n  Per-trial rows written to {out}")


if __name__ == "__main__":
    main()