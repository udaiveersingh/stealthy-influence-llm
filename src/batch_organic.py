"""
Step 5: Run organic trials at volume...
"""
import glob
import json
import os
import statistics as stats

from groq import RateLimitError
from src.llm_client import LLMClient
from src.organic_pipeline import load_agents, load_topic, run_organic_trial, save_trial

N_TRIALS_PER_TOPIC = 5
TOPICS = ["remote_work", "ai_regulation", "college_value"]
NUM_ROUNDS = 3
LOG_DIR = "logs"


def get_completed_seeds(topic_id, condition="organic"):
    """Scan logs/ for trials already completed for this topic, by seed."""
    completed = set()
    pattern = os.path.join(LOG_DIR, f"*__{condition}__{topic_id}__seed*.json")
    for path in glob.glob(pattern):
        with open(path) as f:
            data = json.load(f)
        completed.add(data["seed"])
    return completed


def find_log_path_for(topic_id, seed, condition="organic"):
    """After save_trial() writes a file, locate it by topic+seed so we can report it."""
    pattern = os.path.join(LOG_DIR, f"*__{condition}__{topic_id}__seed{seed:04d}__*.json")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def main():
    client = LLMClient()
    agents = load_agents()

    all_results = {}
    rate_limited = False
    new_files_this_run = []  # (topic_id, seed, avg_shift, filepath) created in THIS execution

    for topic_id in TOPICS:
        if rate_limited:
            break
        topic = load_topic(topic_id)
        already_done = get_completed_seeds(topic_id)
        shifts = []
        print(f"\n{'#'*70}")
        print(f"# TOPIC: {topic_id}")
        print(f"{'#'*70}")
        if already_done:
            print(f"  (found {len(already_done)} completed trial(s) already: seeds {sorted(already_done)})")

        for i in range(N_TRIALS_PER_TOPIC):
            seed = 1000 + i
            if seed in already_done:
                print(f"\n-- Trial {i+1}/{N_TRIALS_PER_TOPIC} (seed={seed}) -- SKIPPED (already done)")
                continue

            print(f"\n-- Trial {i+1}/{N_TRIALS_PER_TOPIC} (seed={seed}) --")
            try:
                trial = run_organic_trial(client, agents, topic, num_rounds=NUM_ROUNDS, seed=seed)
            except RateLimitError as e:
                print(f"\n!! Rate limit hit during {topic_id} trial {i+1}. Stopping here.")
                print(f"   Everything completed so far has been saved. Error: {e}")
                rate_limited = True
                break

            trial_shifts = [
                trial.post_stances[a] - trial.pre_stances[a]
                for a in trial.pre_stances
                if a in trial.post_stances
            ]
            avg_shift = sum(trial_shifts) / len(trial_shifts)
            shifts.append(avg_shift)
            save_trial(trial)
            print(f"   avg shift = {avg_shift:+.3f}")

            # Record exactly which file this trial produced, for the end-of-run report.
            new_path = find_log_path_for(topic_id, seed)
            new_files_this_run.append((topic_id, seed, avg_shift, new_path))

        # Pull ALL completed shifts for this topic (old + new) for the summary.
        all_shifts_this_topic = []
        for path in glob.glob(os.path.join(LOG_DIR, f"*__organic__{topic_id}__seed*.json")):
            with open(path) as f:
                data = json.load(f)
            trial_shifts = [
                data["post_stances"][a] - data["pre_stances"][a]
                for a in data["pre_stances"]
                if a in data["post_stances"]
            ]
            all_shifts_this_topic.append(sum(trial_shifts) / len(trial_shifts))

        if all_shifts_this_topic:
            all_results[topic_id] = all_shifts_this_topic

    # --- Summary ---
    print(f"\n{'='*70}")
    print("STEP 5 SUMMARY: organic stance-shift distribution per topic")
    if rate_limited:
        print("(NOTE: run was cut short by rate limiting -- this summary covers")
        print(" only the trials completed so far, across all runs.)")
    print(f"{'='*70}")
    for topic_id, shifts in all_results.items():
        mean = stats.mean(shifts)
        stdev = stats.stdev(shifts) if len(shifts) > 1 else 0.0
        print(f"\n{topic_id}:")
        print(f"  trials: {[f'{s:+.2f}' for s in shifts]}")
        print(f"  mean = {mean:+.3f}   std dev = {stdev:.3f}   range = [{min(shifts):+.2f}, {max(shifts):+.2f}]")

    # --- NEW: report exactly which files this execution created ---
    print(f"\n{'='*70}")
    print(f"FILES CREATED IN THIS RUN ({len(new_files_this_run)} new trial(s)):")
    print(f"{'='*70}")
    if not new_files_this_run:
        print("  (none -- everything needed was already in logs/)")
    else:
        for topic_id, seed, avg_shift, path in new_files_this_run:
            fname = os.path.basename(path) if path else "!! FILE NOT FOUND (check save_trial naming) !!"
            print(f"  [{topic_id:15s} seed={seed}] shift={avg_shift:+.3f}  ->  {fname}")

    print(f"\n{'='*70}")
    if rate_limited:
        print("  - Re-run this script once your Groq limit resets.")
        print("    It will SKIP already-completed seeds and continue where it left off.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()