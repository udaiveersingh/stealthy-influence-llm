"""
Find and pretty-print a specific trial by topic + seed, so you don't have to
manually hunt through filenames (which use UUIDs, not seeds).

Usage:
    python -m src.find_trial <topic_id> <seed>

Example:
    python -m src.find_trial centralized_governance 2004
"""
import glob
import json
import os
import sys

LOG_DIR = "logs"


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.find_trial <topic_id> <seed>")
        sys.exit(1)

    topic_id = sys.argv[1]
    seed = int(sys.argv[2])

    for path in glob.glob(os.path.join(LOG_DIR, "*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("topic_id") == topic_id and data.get("seed") == seed and data.get("condition") == "organic":
            print(f"FOUND: {path}\n")
            print(json.dumps(data, indent=2))
            return

    print(f"No organic trial found for topic={topic_id}, seed={seed}")


if __name__ == "__main__":
    main()
