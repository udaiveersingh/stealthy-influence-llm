"""
Shows the FULL content of flagged messages (not audit_logs.py's 120-char
display preview) and exactly which individual check fired.

Why this exists: audit_logs.py found an 8.9% flag rate across the full log
set, versus ~0.1% in an earlier, smaller audit. Before treating that as real,
we need to see whole messages -- a 120-char preview cut mid-word looks
"truncated" regardless of whether the real, full message ends properly.

Usage:
    python inspect_flagged.py logs/*.json                  # first 15 flagged, full text
    python inspect_flagged.py logs/*.json --reason other    # only the 'other' bucket
    python inspect_flagged.py logs/*.json --limit 30
"""
import argparse
import glob
import json

from src.conversation import (
    _is_placeholder_leak, _is_gibberish, _has_excessive_repetition,
    _has_intraword_repetition, _looks_truncated, _has_glued_words,
    _META_COMMENTARY_PHRASES, MIN_POST_WORDS,
)


def which_checks_fire(text: str) -> list:
    """Unlike audit_logs.py's 3-bucket classify(), this names every
    individual check that fires, so 'other' stops being a black box."""
    fired = []
    lowered = text.lower()
    if "<your message>" in lowered or "<my message>" in lowered:
        fired.append("literal_template_leak")
    if len(text.strip().split()) < MIN_POST_WORDS:
        fired.append(f"too_short(<{MIN_POST_WORDS}_words)")
    if any(p in lowered for p in _META_COMMENTARY_PHRASES):
        fired.append("meta_commentary_phrase")
    if _is_gibberish(text):
        fired.append("gibberish")
    if _has_excessive_repetition(text):
        fired.append("excessive_repetition")
    if _has_intraword_repetition(text):
        fired.append("intraword_repetition")
    if _looks_truncated(text):
        fired.append("looks_truncated")
    if _has_glued_words(text):
        fired.append("glued_words")
    return fired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    paths = []
    for p in args.paths:
        paths.extend(glob.glob(p))

    shown = 0
    for path in sorted(set(paths)):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for m in data.get("messages", []):
            content = m.get("content", "")
            if not _is_placeholder_leak(content):
                continue
            checks = which_checks_fire(content)
            print(f"\n{'='*90}")
            print(f"{path} | round {m.get('round_number')} | {m.get('agent_id')}")
            print(f"checks fired: {checks}")
            print(f"length: {len(content)} chars, {len(content.split())} words")
            print(f"FULL TEXT:\n{content}")
            print(f"LAST 15 CHARS: {content[-15:]!r}")
            shown += 1
            if shown >= args.limit:
                print(f"\n(stopping at --limit {args.limit})")
                return


if __name__ == "__main__":
    main()