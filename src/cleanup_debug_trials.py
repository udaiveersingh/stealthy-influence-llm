"""
One-time cleanup: removes ad-hoc debugging trials (seed=9001, used throughout
this troubleshooting session by test_nvidia_backend.py) from logs/ so they
don't get mixed into real batch statistics.

Only removes organic trials with seed=9001 -- the real batch uses seeds
2000-2004 exclusively, so this is a safe, precise filter.

Usage:
    python -m src.cleanup_debug_trials          # dry run, lists what WOULD be deleted
    python -m src.cleanup_debug_trials --delete # actually deletes
"""
import glob
import json
import os
import sys

LOG_DIR = "logs"
DEBUG_SEED = 9001


def main():
    do_delete = "--delete" in sys.argv
    to_remove = []

    for path in glob.glob(os.path.join(LOG_DIR, "*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("seed") == DEBUG_SEED:
            to_remove.append((path, data.get("model_provider", "?"), data.get("topic_id", "?")))

    if not to_remove:
        print("No seed=9001 debug trials found -- nothing to clean up.")
        return

    print(f"Found {len(to_remove)} debug trial(s) with seed={DEBUG_SEED}:\n")
    for path, provider, topic in to_remove:
        print(f"  {os.path.basename(path)}  (provider={provider}, topic={topic})")

    if do_delete:
        for path, _, _ in to_remove:
            os.remove(path)
        print(f"\nDeleted {len(to_remove)} file(s).")
    else:
        print(f"\nDRY RUN -- nothing deleted. Re-run with --delete to actually remove these.")


if __name__ == "__main__":
    main()
