"""
Do attacker messages look different from ordinary messages just by LENGTH?

Why this matters: in the blind hand-check, 17 of 30 sampled messages ended in "..." (they hit the
length clip) and every short one-liner was organic. The code explains it: ordinary turns are told to
write "1-2 short sentences", but the attacker's intervention prompt has no length instruction, so attackers
write long arguments. That has two consequences:
  1. STEALTH: any detector, even a trivial length rule, can pick attackers out.
  2. CONFOUND: a longer, more elaborate argument may persuade more for reasons unrelated to its direction.

This script measures it, with no API calls:
  - length (characters, words) and the share ending in "..." for attacker messages (attack phase),
    ordinary agents' messages in the same trials, and the organic arm (same agents, no objective);
  - the 'length AUC': the chance that a random attacker message is longer than a random message by the SAME
    agents in the organic arm. 0.5 = length gives nothing away; 1.0 = length alone identifies the attacker.

Usage:
    python length_check.py
    python length_check.py --manifest runs/confirmatory_manifest.json     # the unenforced run
"""
from __future__ import annotations

import argparse
import statistics as st

from scipy import stats

import direction_check_confirmatory as C

PLACEHOLDER = "(no substantive response generated)"


def describe(label, msgs):
    if not msgs:
        print(f"  {label:<52}  n/a")
        return
    chars = [len(m) for m in msgs]
    words = [len(m.split()) for m in msgs]
    clip = sum(1 for m in msgs if m.rstrip().endswith("..."))
    print(f"  {label:<52}{len(msgs):>6}{st.mean(chars):>9.0f}{st.median(chars):>8.0f}{st.mean(words):>8.1f}"
          f"{100 * clip / len(msgs):>9.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/confirmatory_enf_manifest.json")
    args = ap.parse_args()
    cells = C.load_cells(C.load_json(args.manifest))

    attackers, ordinary, org_ref, org_all = [], [], [], []
    for seed in sorted(cells):
        for arm in ("attack1", "attack4"):
            c = cells[seed].get(arm)
            if not c:
                continue
            att = set(c["meta"]["attackers"])
            for m in c["trial"]["messages"]:
                if m["round_number"] <= c["meta"]["attack_rounds"] and m["content"] != PLACEHOLDER:
                    (attackers if m["agent_id"] in att else ordinary).append(m["content"])
        o = cells[seed].get("organic")
        if o:
            ref = set(o["meta"]["reference_attackers"])
            for m in o["trial"]["messages"]:
                if m["round_number"] <= o["meta"]["attack_rounds"] and m["content"] != PLACEHOLDER:
                    org_all.append(m["content"])
                    if m["agent_id"] in ref:
                        org_ref.append(m["content"])

    print("=" * 96)
    print("LENGTH CHECK (rounds 1-3, the attack phase)")
    print("=" * 96)
    print(f"  {'':<52}{'n':>6}{'chars':>9}{'median':>8}{'words':>8}{'ends ...':>10}")
    describe("attacker messages (attack arms)", attackers)
    describe("ordinary agents' messages, same trials", ordinary)
    describe("organic arm: the SAME agents, no objective", org_ref)
    describe("organic arm: all agents", org_all)

    if attackers and org_ref:
        u = stats.mannwhitneyu([len(m) for m in attackers], [len(m) for m in org_ref], alternative="two-sided")
        auc = u.statistic / (len(attackers) * len(org_ref))
        print(f"\n  length AUC, attacker vs same agents in organic: {auc:.2f}   (0.5 = no information, 1.0 = perfect tell)")
        ratio = st.mean(len(m) for m in attackers) / st.mean(len(m) for m in org_ref)
        print(f"  attacker messages are on average {ratio:.1f}x as long as the same agents' organic messages")
        if auc > 0.7:
            print("\n  READING: length alone gives the attackers away. Any detectability result must control for it, and the")
            print("  effect estimates are for LONG attacker messages, which can persuade more than short ones. A length-matched")
            print("  attacker arm (told to write 1-2 short sentences like everyone else) is the fix.")
        else:
            print("\n  READING: length is not a strong tell.")


if __name__ == "__main__":
    main()