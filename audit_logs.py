"""
Audits already-collected trial JSON logs (TrialRecord.to_dict() output) for
messages that would have been caught by the NEW, stronger leak detector in
conversation.py but weren't caught by the version that was active when your
existing organic baseline was collected.

This matters because the corruption pattern seen in the Step 6 pilot
(garbled ASCII text like '...aminoells overtellsells inolesells...') hit an
ORDINARY agent, not the adversary -- meaning it's a defect in message
generation itself, and could be silently present in your existing organic
dataset too, since the old detector only checked for literal template leaks
and a minimum word count.

Usage:
    python audit_logs.py logs/*.json
    python audit_logs.py logs/some_specific_trial.json
"""
from __future__ import annotations

import glob
import json
import sys

from src.conversation import _is_placeholder_leak, _is_gibberish, _has_excessive_repetition


def classify(text: str) -> str:
    if _has_excessive_repetition(text):
        return "excessive_repetition (decoding-glitch corruption)"
    if _is_gibberish(text):
        return "gibberish (non-Latin / low-alpha-ratio)"
    return "other (short/meta-commentary/template leak)"


def audit_file(path: str):
    with open(path) as f:
        data = json.load(f)

    messages = data.get("messages", [])
    trial_id = data.get("trial_id", path)
    flagged = []
    for m in messages:
        content = m.get("content", "")
        if _is_placeholder_leak(content):
            flagged.append((m.get("agent_id"), m.get("round_number"), classify(content), content))

    if flagged:
        print(f"\n{trial_id} ({path}) -- {len(flagged)}/{len(messages)} messages flagged:")
        for agent_id, round_num, reason, content in flagged:
            preview = content[:120] + ("..." if len(content) > 120 else "")
            print(f"  [round {round_num}] {agent_id} -- {reason}\n    {preview!r}")
    return len(flagged), len(messages)


def main():
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))
    if not paths:
        print("Usage: python audit_logs.py <path-or-glob> [more paths...]")
        print("Example: python audit_logs.py logs/*.json")
        return

    total_flagged = 0
    total_messages = 0
    files_with_issues = 0
    for path in sorted(set(paths)):
        try:
            n_flagged, n_messages = audit_file(path)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {path}: {e}")
            continue
        total_flagged += n_flagged
        total_messages += n_messages
        if n_flagged:
            files_with_issues += 1

    print(f"\n--- Summary ---")
    print(f"Files scanned: {len(paths)}")
    print(f"Files with at least one flagged message: {files_with_issues}")
    print(f"Total flagged messages: {total_flagged} / {total_messages}")
    if total_messages:
        print(f"Flagged rate: {total_flagged / total_messages:.1%}")


if __name__ == "__main__":
    main()
