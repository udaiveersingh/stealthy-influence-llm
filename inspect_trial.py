"""
Inspect one seed of the confirmatory study: is a surprising number real or an artifact?

    python inspect_trial.py --seed 2 --arm attack4

Prints, for that seed and arm: every agent's stance at the three readings (with role), the
same agents' values in the organic arm of that seed, flags for extreme stance values, then the
attackers' messages and what the measured agents said in rounds 3 and 6 (the rounds just
before each stance reading).
"""
from __future__ import annotations

import argparse
import json

MANIFEST = "runs/confirmatory_manifest.json"


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", default="attack4", choices=["organic", "attack1", "attack4"])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--width", type=int, default=420, help="max characters shown per message")
    args = ap.parse_args()

    m = load(args.manifest)["completed"]
    e = next((v for v in m.values() if v["seed"] == args.seed and v["arm"] == args.arm), None)
    eo = next((v for v in m.values() if v["seed"] == args.seed and v["arm"] == "organic"), None)
    if not e or not eo:
        raise SystemExit(f"seed {args.seed} / arm {args.arm} (or its organic arm) not found")
    t, meta = load(e["path"]), load(e["meta_path"])
    org = load(eo["path"])
    sign = 1 if meta["target_stance"] >= 0 else -1
    attackers, measured = set(meta["attackers"]), set(meta["measured_ids"])

    print("=" * 96)
    print(f"seed {args.seed}, arm {args.arm}: attackers {sorted(attackers) or 'none'}; "
          f"measured {sorted(measured)}; target {meta['target_stance']:+.2f}")
    print("=" * 96)
    print(f"{'agent':<14}{'role':<10}{'pre':>7}{'end':>7}{'final':>7}   {'toward (end)':>13}  |  organic: {'end':>6}{'final':>7}  flags")
    for a in sorted(t["pre_stances"]):
        role = "ATTACKER" if a in attackers else ("measured" if a in measured else "other")
        pre, end, fin = t["pre_stances"][a], t["post_stances"][a], t["post_removal_stances"][a]
        oe, of = org["post_stances"][a], org["post_removal_stances"][a]
        flags = []
        for name, v in (("pre", pre), ("end", end), ("final", fin)):
            if abs(v) >= 0.999:
                flags.append(f"{name}=EXTREME")
        if abs(end - pre) >= 0.8:
            flags.append("moved>=0.8")
        print(f"{a:<14}{role:<10}{pre:>+7.2f}{end:>+7.2f}{fin:>+7.2f}   {sign * (end - pre):>+13.2f}  |           "
              f"{oe:>+6.2f}{of:>+7.2f}  {' '.join(flags)}")
    ms = [sign * (t["post_stances"][a] - t["pre_stances"][a]) for a in measured]
    print(f"\nmean toward-shift of measured agents at end of attack: {sum(ms) / len(ms):+.3f}  "
          f"(individual: {', '.join(f'{x:+.2f}' for x in ms)})")

    def show(round_number, who, label):
        print(f"\n--- {label} ---")
        for x in t["messages"]:
            if x["round_number"] == round_number and who(x["agent_id"]):
                tag = "ATTACKER" if x["agent_id"] in attackers else "        "
                print(f"[{x['agent_id'].split('_')[0]:<3}{tag}] {x['content'][:args.width]}")

    for r in (1, 2, 3):
        show(r, lambda a: a in attackers, f"round {r}: attacker messages")
    show(3, lambda a: a in measured, "round 3: what the MEASURED agents said just before the end-of-attack reading")
    show(6, lambda a: a in measured, "round 6: what the MEASURED agents said just before the final reading")


if __name__ == "__main__":
    main()