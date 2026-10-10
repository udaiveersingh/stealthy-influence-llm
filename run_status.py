"""
What finished, what is damaged, what to run next. FREE: reads files only, no API calls.

Use after a crash / closed terminal. The manifests are the record of what completed (written atomically after each
trial), so this does not need your terminal history.

  python run_status.py                 # report + exact resume commands
  python run_status.py --quarantine    # also move orphan log files (no manifest entry) into logs/_orphans/

A trial is DAMAGED if: its log file is missing/unreadable, it does not have 48 messages (8 agents x 6 rounds),
or it lacks the post-attack / final stance readings. Damaged trials are listed with the key to delete from the
manifest so a re-run regenerates them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

EXPECTED_MESSAGES = 48
DEFAULT_TOPIC = "ai_regulation"
EXPECTED_SEEDS = {"ai_regulation": 20}
OTHER_SEEDS = 10
TOPICS = ["ai_regulation", "remote_work", "college_value", "free_speech_harm",
          "traditional_family_roles", "religion_public_life", "centralized_governance", "individual_vs_collective"]

# (label, manifest base path, log dir base, arms expected, CLI flags)
KINDS = [
    ("unenforced", "runs/confirmatory_manifest.json", "logs/confirmatory", ["organic", "attack1", "attack4"], ""),
    ("enforced", "runs/confirmatory_enf_manifest.json", "logs/confirmatory_enf", ["organic", "attack1", "attack4"], "--enforce-direction"),
    ("short", "runs/confirmatory_short_manifest.json", "logs/confirmatory_short", ["organic", "attack4"], "--match-length --enforce-direction"),
    ("diverse", "runs/confirmatory_diverse_manifest.json", "logs/confirmatory_diverse", ["organic", "attack4"], "--diversify-angles --match-length --enforce-direction"),
]


def topic_path(path, topic):
    if topic == DEFAULT_TOPIC:
        return path
    return path.replace(".json", f"__{topic}.json") if path.endswith(".json") else f"{path}__{topic}"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def check_entry(e):
    """Return a problem string, or None if the trial looks complete."""
    t = load(e.get("path", ""))
    if t is None:
        return "log file missing or unreadable"
    if not os.path.exists(e.get("meta_path", "")):
        return "meta file missing"
    n = len(t.get("messages", []))
    if n != EXPECTED_MESSAGES:
        return f"{n} messages (expected {EXPECTED_MESSAGES})"
    if not t.get("post_stances"):
        return "no end-of-attack stance reading"
    if not t.get("post_removal_stances"):
        return "no final stance reading"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarantine", action="store_true", help="move orphan log files to logs/_orphans/")
    args = ap.parse_args()

    resume = []
    damaged_all = []
    print("=" * 100)
    print("RUN STATUS (reads manifests and log files only; no API calls)")
    print("=" * 100)

    for label, mpath, ldir, arms, flags in KINDS:
        rows = []
        for topic in TOPICS:
            mp = topic_path(mpath, topic)
            if not os.path.exists(mp):
                continue
            m = load(mp)
            if m is None:
                print(f"  !! {mp} exists but cannot be read (corrupt?). Restore from git or the .tmp file next to it.")
                continue
            want = EXPECTED_SEEDS.get(topic, OTHER_SEEDS)
            done = {a: set() for a in arms}
            bad = []
            for key, e in m.get("completed", {}).items():
                arm = e.get("arm")
                problem = check_entry(e)
                if problem:
                    bad.append((key, problem, mp))
                elif arm in done:
                    done[arm].add(e["seed"])
            rows.append((topic, want, done, bad, ldir))
            damaged_all.extend(bad)

            # resume commands for anything short of the target
            for arm in arms:
                missing = [s for s in range(1, want + 1) if s not in done[arm]]
                if missing and not (label != "unenforced" and arm == "organic"):
                    # enforced/short/diverse reuse organic from the unenforced manifest, so only the main arm is run here
                    pass
            main_arms = [a for a in arms if a != "organic"] if label != "unenforced" else ["organic"]
            for arm in main_arms:
                missing = [s for s in range(1, want + 1) if s not in done[arm]]
                if not missing:
                    continue
                tflag = "" if topic == DEFAULT_TOPIC else f"--topic {topic} --target auto "
                arm_flag = f"--arms {arm} " if label == "unenforced" else (f"--arms {' '.join(main_arms)} " if label == "enforced" else "")
                resume.append((label, topic, arm, missing,
                               f"python run_confirmatory.py {tflag}{flags} {arm_flag}--seeds {' '.join(map(str, missing))}".replace("  ", " ")))

        if not rows:
            continue
        print(f"\n[{label}]  manifest base: {mpath}")
        print(f"  {'topic':<26}" + "".join(f"{a:>10}" for a in arms) + "   damaged")
        for topic, want, done, bad, _ in rows:
            cells = "".join(f"{len(done[a]):>6}/{want:<3}" for a in arms)
            print(f"  {topic:<26}{cells}   {len(bad)}")

    # damaged entries
    print("\n" + "-" * 100)
    if damaged_all:
        print(f"DAMAGED ENTRIES ({len(damaged_all)}): fix before analysing")
        for key, problem, mp in damaged_all:
            print(f"  {mp}: {key}  -> {problem}")
        print("  To regenerate: delete that key from the manifest's \"completed\" dict (or ask me for a repair script), then re-run.")
    else:
        print("No damaged entries: every manifest entry has a readable log with 48 messages and all three stance readings.")

    # orphans: log files on disk that no manifest points at
    orphans = []
    for label, mpath, ldir, arms, flags in KINDS:
        for topic in TOPICS:
            mp = topic_path(mpath, topic)
            ld = topic_path(ldir, topic)
            if not os.path.isdir(ld):
                continue
            m = load(mp) if os.path.exists(mp) else {"completed": {}}
            known = set()
            for e in (m or {}).get("completed", {}).values():
                known.add(os.path.abspath(e.get("path", "")))
                known.add(os.path.abspath(e.get("meta_path", "")))
            for f in glob.glob(os.path.join(ld, "*.json")):
                if os.path.abspath(f) not in known:
                    orphans.append(f)
    print("\n" + "-" * 100)
    if orphans:
        print(f"ORPHAN LOG FILES ({len(orphans)}): on disk but not in any manifest (a trial that was killed before it was recorded).")
        print("They are NOT used by any analysis. A re-run makes a fresh trial. Safe to leave; --quarantine moves them aside.")
        for f in orphans[:20]:
            print("  ", f)
        if len(orphans) > 20:
            print(f"   ... and {len(orphans) - 20} more")
        if args.quarantine:
            os.makedirs("logs/_orphans", exist_ok=True)
            for f in orphans:
                shutil.move(f, os.path.join("logs/_orphans", os.path.basename(f)))
            print(f"  moved {len(orphans)} files to logs/_orphans/")
    else:
        print("No orphan log files.")

    # resume plan
    print("\n" + "=" * 100)
    if resume:
        print("STILL TO RUN (copy these; set NVIDIA_API_KEY and LLM_MAX_RPM in the SAME window first; run one at a time):")
        for label, topic, arm, missing, cmd in resume:
            print(f"\n  [{label} | {topic} | {arm}] missing seeds {missing}")
            print(f"    {cmd}")
        print("\n  Order matters: finish the unenforced/organic runs for a topic before its enforced/short/diverse runs.")
    else:
        print("Everything in the expected set is complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()