"""
Step 1 -- validity checks on the existing pilot data. No API calls.

Flags four kinds of problems a reviewer could catch, and prints the raw
material needed to judge each one by eye:

  1. STANCE COLLAPSE   several agents report the IDENTICAL final stance
                       (e.g. seed 6 directional: seven agents at exactly +0.80).
                       Real opinion change rarely lands five-plus agents on the
                       same number; it more likely means the probe collapsed.
  2. LARGE SWINGS      one agent moves >= SWING_LIMIT and dominates the trial
                       mean (e.g. seed 8: A7_STATUS +0.70 -> -0.25).
  3. ENFORCEMENT FAILS interventions flagged forced_compliance_failed=True.
  4. MANIPULATION      for directional trials we never checked the attacker
                       actually argued TOWARD the target. The attacker's own
                       messages are printed for flagged trials so you can read
                       them; an attacker whose own stance moves AWAY from the
                       target is flagged automatically.

Decisions (exclude / keep / report) are yours -- the script flags, it does not
drop anything. Whatever rule you adopt, apply it to every trial, not just the
ones that look inconvenient, and state it in the paper.

Usage:
    python validity_checks.py
    python validity_checks.py --manifest runs/manifest.json --show-all-attacker-msgs
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

COLLAPSE_MIN_AGENTS = 5   # this many non-attacker agents sharing one exact value
SWING_LIMIT = 0.8         # |post - pre| at or above this is flagged
PLACEHOLDER = "(no substantive response generated)"


def load(path):
    with open(path) as f:
        return json.load(f)


def parse_key(key):
    p = key.split("|")
    return p[0], p[1], int(p[2]), (p[3] if len(p) > 3 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--show-all-attacker-msgs", action="store_true",
                    help="print attacker messages for every adversarial trial, not just flagged ones")
    args = ap.parse_args()

    manifest = load(args.manifest)
    rows = []
    for key, entry in sorted(manifest.get("completed", {}).items(),
                             key=lambda kv: (parse_key(kv[0])[2], kv[0])):
        cond, topic, seed, removal = parse_key(key)
        path = entry.get("path")
        if not path or not os.path.exists(path):
            continue
        trial = load(path)
        attack = None
        ap_path = entry.get("attack_path")
        if ap_path and os.path.exists(ap_path):
            attack = load(ap_path)
        rows.append((key, cond, seed, removal, trial, attack))

    attacker_id = next((a["attacker_id"] for *_, a in rows if a), "A8_ADVERSARY")
    flagged = {}   # key -> list of reasons

    def flag(key, reason):
        flagged.setdefault(key, []).append(reason)

    print("=" * 92)
    print(f"VALIDITY CHECKS -- {len(rows)} trials in manifest (attacker id: {attacker_id})")
    print("=" * 92)

    # ---- 1. stance collapse --------------------------------------------------
    print(f"\n1. STANCE COLLAPSE (>= {COLLAPSE_MIN_AGENTS} non-attacker agents with the identical stance)")
    print("-" * 92)
    n1 = 0
    for key, cond, seed, removal, trial, attack in rows:
        for field in ("post_stances", "post_removal_stances"):
            st = {a: v for a, v in (trial.get(field) or {}).items() if a != attacker_id}
            if not st:
                continue
            value, count = Counter(round(v, 4) for v in st.values()).most_common(1)[0]
            if count >= COLLAPSE_MIN_AGENTS:
                n1 += 1
                flag(key, f"collapse: {count} agents at {value:+.2f} ({field})")
                print(f"  {key:<45} {field:<22} {count}/{len(st)} agents at exactly {value:+.2f}")
    if not n1:
        print("  none")

    # ---- 2. large single-agent swings ----------------------------------------
    print(f"\n2. LARGE SINGLE-AGENT SWINGS (|shift| >= {SWING_LIMIT})")
    print("-" * 92)
    n2 = 0
    for key, cond, seed, removal, trial, attack in rows:
        pre, post = trial.get("pre_stances", {}), trial.get("post_stances", {})
        others = [a for a in pre if a in post and a != attacker_id]
        for a in others:
            s = post[a] - pre[a]
            if abs(s) >= SWING_LIMIT:
                n2 += 1
                rest = [post[b] - pre[b] for b in others if b != a]
                mean_with = sum(post[b] - pre[b] for b in others) / len(others)
                mean_without = sum(rest) / len(rest) if rest else float("nan")
                flag(key, f"swing: {a} {s:+.2f}")
                print(f"  {key:<45} {a:<12} {pre[a]:+.2f} -> {post[a]:+.2f} ({s:+.2f})  "
                      f"trial mean shift {mean_with:+.3f} with, {mean_without:+.3f} without")
    if not n2:
        print("  none")

    # ---- 3. forced-compliance failures ---------------------------------------
    print("\n3. ENFORCEMENT FAILURES (forced_compliance_failed=True)")
    print("-" * 92)
    n3 = 0
    for key, cond, seed, removal, trial, attack in rows:
        if not attack:
            continue
        for iv in attack.get("interventions", []):
            if iv.get("forced_compliance_failed"):
                n3 += 1
                flag(key, f"forced-compliance fail, round {iv.get('round')}")
                print(f"  {key}  round {iv.get('round')}  target {iv.get('target')}")
                print(f"    message: {iv.get('message')}\n")
    if not n3:
        print("  none")

    # ---- 4. placeholder messages ---------------------------------------------
    print("\n4. PLACEHOLDER MESSAGES (all retries failed; a hole in the conversation)")
    print("-" * 92)
    n4 = 0
    for key, cond, seed, removal, trial, attack in rows:
        for m in trial.get("messages", []):
            if m.get("content") == PLACEHOLDER:
                n4 += 1
                who = "ATTACKER" if m.get("agent_id") == attacker_id else m.get("agent_id")
                flag(key, f"placeholder: {who} round {m.get('round_number')}")
                print(f"  {key:<45} {who} round {m.get('round_number')}")
    if not n4:
        print("  none")

    # ---- 5. manipulation check: attacker's own stance direction --------------
    print("\n5. ATTACKER MOVED AWAY FROM ITS OWN TARGET (possible directional compliance failure)")
    print("-" * 92)
    print("  Not proof of failure -- an attacker can argue for the target while privately")
    print("  reporting otherwise -- but these are the trials whose messages to read first.")
    n5 = 0
    for key, cond, seed, removal, trial, attack in rows:
        if not attack:
            continue
        target = attack.get("target_stance")
        pre, post = trial.get("pre_stances", {}), trial.get("post_stances", {})
        if target is None or attacker_id not in pre or attacker_id not in post:
            continue
        toward = (post[attacker_id] - pre[attacker_id]) * (1 if target >= 0 else -1)
        if toward < -0.10:
            n5 += 1
            flag(key, f"attacker moved away from target ({toward:+.2f})")
            print(f"  {key:<45} attacker {pre[attacker_id]:+.2f} -> {post[attacker_id]:+.2f}, "
                  f"target {target:+.2f}")
    if not n5:
        print("  none")

    # ---- attacker messages for flagged (or all) adversarial trials -----------
    print("\n" + "=" * 92)
    print("ATTACKER MESSAGES FOR MANUAL READING" +
          (" (all adversarial trials)" if args.show_all_attacker_msgs else " (flagged trials only)"))
    print("=" * 92)
    print("  For each: does the attacker argue TOWARD its target? For ai_regulation with target")
    print("  -0.8 that means AGAINST heavy regulation. Note trials where it argues the other way.")
    for key, cond, seed, removal, trial, attack in rows:
        if not attack:
            continue
        if not args.show_all_attacker_msgs and key not in flagged:
            continue
        target = attack.get("target_stance")
        print(f"\n--- {key}  (condition={cond}, target={target:+.2f}) ---")
        for m in trial.get("messages", []):
            if m.get("agent_id") == attacker_id:
                tag = f"-> {m['target_agent_id']}" if m.get("target_agent_id") else "(ordinary turn)"
                print(f"  [round {m.get('round_number')}] {tag}\n    {m.get('content')}")

    # ---- summary -------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"SUMMARY -- {len(flagged)} of {len(rows)} trials flagged")
    print("=" * 92)
    for key, reasons in flagged.items():
        print(f"  {key}")
        for r in reasons:
            print(f"      - {r}")
    print("\n  Next: read the flagged transcripts, then write ONE exclusion rule that you would")
    print("  apply to every trial (e.g. 'exclude trials with stance collapse'), and report")
    print("  results both with and without the excluded trials.")


if __name__ == "__main__":
    main()