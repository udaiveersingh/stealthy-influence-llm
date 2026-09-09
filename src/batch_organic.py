"""
Batch organic trial runner.

CHANGES in this version:
- Fresh seed range (2000+) to avoid any collision with the old seed 1000-1004
  set that got accidentally double-run under two model configs previously.
- Completion-checking now also matches on model_provider, so gpt-oss and
  Nemotron runs never get confused with each other or silently merged.
- Records model_provider/model_name in every trial (via TrialRecord, already
  stamped by run_organic_trial).
- Uses the now-8-agent-by-default population for organic/adversarial parity.

Usage:
    $env:MODEL_PROVIDER="groq"     # or "nvidia"
    python -m src.batch_organic
"""
import glob
import json
import os
import statistics as stats

from src.llm_client import LLMClient
from src.organic_pipeline import load_agents, load_topic, run_organic_trial, save_trial

N_TRIALS_PER_TOPIC = 5
TOPICS = [
    "remote_work",               # social_lifestyle
    "ai_regulation",              # technology_policy
    "college_value",              # education
    "free_speech_harm",           # human_values
    "traditional_family_roles",   # relationships_society
    "religion_public_life",       # religion_belief
    "centralized_governance",     # politics_governance
    "individual_vs_collective",   # ethics_morality
]  # Phase 3: all 8 categories per professor's request, now that backend is validated
NUM_ROUNDS = 3
SEED_BASE = 2000  # fresh range, avoids any collision with prior runs


def _load_all_logs(condition="organic"):
    records = []
    for path in glob.glob(os.path.join("logs", "*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("condition") == condition:
            data["_filepath"] = path
            records.append(data)
    return records


def get_completed_seeds(topic_id, model_provider, condition="organic"):
    """Content-based check, now ALSO matching on model_provider -- so a Nemotron
    run and a gpt-oss run with the same seed never look like duplicates of
    each other."""
    return {
        r["seed"] for r in _load_all_logs(condition)
        if r.get("topic_id") == topic_id and r.get("model_provider") == model_provider
    }


def get_all_shifts_for_topic(topic_id, model_provider, condition="organic"):
    shifts = []
    for r in _load_all_logs(condition):
        if r.get("topic_id") != topic_id or r.get("model_provider") != model_provider:
            continue
        trial_shifts = [
            r["post_stances"][a] - r["pre_stances"][a]
            for a in r["pre_stances"] if a in r["post_stances"]
        ]
        if trial_shifts:
            shifts.append(sum(trial_shifts) / len(trial_shifts))
    return shifts


def main():
    client = LLMClient()
    model_info = client.info()
    print(f"Running with provider={model_info['provider']}  model={model_info['model']}\n")

    agents = load_agents()  # 8 agents, parity-correct
    print(f"Agent count: {len(agents)} (should be 8)\n")

    all_results = {}
    new_files_this_run = []

    for topic_id in TOPICS:
        topic = load_topic(topic_id)
        already_done = get_completed_seeds(topic_id, model_info["provider"])
        print(f"\n{'#'*70}")
        print(f"# TOPIC: {topic_id}  (provider={model_info['provider']})")
        print(f"{'#'*70}")
        if already_done:
            print(f"  (found {len(already_done)} completed trial(s) already: seeds {sorted(already_done)})")

        for i in range(N_TRIALS_PER_TOPIC):
            seed = SEED_BASE + i
            if seed in already_done:
                print(f"\n-- Trial {i+1}/{N_TRIALS_PER_TOPIC} (seed={seed}) -- SKIPPED (already done)")
                continue

            print(f"\n-- Trial {i+1}/{N_TRIALS_PER_TOPIC} (seed={seed}) --")
            try:
                trial = run_organic_trial(client, agents, topic, num_rounds=NUM_ROUNDS, seed=seed)
            except Exception as e:
                # With 8 topics x 5 trials = 40 trials in one run, don't let a
                # single unexpected failure (e.g. exhausted retries) kill the
                # whole batch -- log it clearly and move on to the next trial.
                print(f"   !! FAILED (skipping this trial): {e}")
                continue

            trial_shifts = [
                trial.post_stances[a] - trial.pre_stances[a]
                for a in trial.pre_stances if a in trial.post_stances
            ]
            avg_shift = sum(trial_shifts) / len(trial_shifts)
            saved_path = save_trial(trial)
            print(f"   avg shift = {avg_shift:+.3f}")
            new_files_this_run.append((topic_id, seed, avg_shift, saved_path))

        all_shifts = get_all_shifts_for_topic(topic_id, model_info["provider"])
        if all_shifts:
            all_results[topic_id] = all_shifts

    print(f"\n{'='*70}")
    print(f"SUMMARY: organic stance-shift distribution ({model_info['provider']}/{model_info['model']})")
    print(f"{'='*70}")
    for topic_id, shifts in all_results.items():
        mean = stats.mean(shifts)
        stdev = stats.stdev(shifts) if len(shifts) > 1 else 0.0
        print(f"\n{topic_id}:")
        print(f"  trials: {[f'{s:+.2f}' for s in shifts]}")
        print(f"  mean = {mean:+.3f}   std dev = {stdev:.3f}   range = [{min(shifts):+.2f}, {max(shifts):+.2f}]")

    print(f"\n{'='*70}")
    print(f"FILES CREATED IN THIS RUN ({len(new_files_this_run)} new trial(s)):")
    print(f"{'='*70}")
    for topic_id, seed, avg_shift, path in new_files_this_run:
        print(f"  [{topic_id:15s} seed={seed}] shift={avg_shift:+.3f}  ->  {os.path.basename(path)}")
    if not new_files_this_run:
        print("  (none -- everything needed was already in logs/)")


if __name__ == "__main__":
    main()