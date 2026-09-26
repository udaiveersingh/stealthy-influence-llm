"""
Step 6 main data collection.

Design:
  For each (topic, seed):
    1. Run ONE organic control trial (8 agents, A8 behaving ordinarily).
    2. Take its pre_stances as frozen_initial_stances.
    3. Run one adversarial trial per mechanism, all starting from those
       SAME frozen stances.

Resumability: every completed trial writes a marker into runs/manifest.json.

Parallelism (--parallel N): each (topic, seed) "cell" -- its organic control
plus that seed's adversarial trials -- is fully independent of every other
seed's cell. Cells can run concurrently. This only works because:
  - organic_pipeline.run_organic_trial and adversarial_pipeline.run_adversarial_trial
    now use a per-trial random.Random(seed) instance instead of the global
    random.seed()/shuffle -- see the comments in those files. Without that
    fix, concurrent seeds would race on shared RNG state and silently break
    reproducibility.
  - The manifest is protected by a lock: concurrent trials finishing at
    nearly the same time would otherwise read-modify-write runs/manifest.json
    at the same time and could drop each other's updates.

Usage:
    python run_step6.py --estimate
    python run_step6.py                                  # sequential (default)
    python run_step6.py --parallel 4                      # 4 seed-cells at once
    python run_step6.py --topics ai_regulation --seeds 1 2 3
    python run_step6.py --mechanisms disinformation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm_client import LLMClient
from src.adversary import Mechanism, Targeting, Adaptivity, load_disinfo_library
from src.organic_pipeline import load_agents, load_topic, run_organic_trial, save_trial
from src.adversarial_pipeline import (
    run_adversarial_trial,
    save_adversarial_trial,
    summarize_adversarial_trial,
    find_attacker_id,
)

MANIFEST_PATH = "runs/manifest.json"
_manifest_lock = threading.Lock()

DEFAULT_TOPICS = ["ai_regulation"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_MECHANISMS = ["directional", "disinformation", "hedged_disinformation", "manufactured_consensus"]
# personalized_persuasion dropped from the default set: Run 1 showed no
# reliable effect (d=-0.22, p=0.649) and it isn't keyword-checkable, so it
# costs a quarter of the budget for a result already established. Still
# supported via --mechanisms if you want it.

ENFORCE_COMPLIANCE_MECHANISMS = {Mechanism.DISINFORMATION, Mechanism.HEDGED_DISINFORMATION}
# Both check the SAME claim keywords, so enforcing both isolates "how
# directly the claim is asserted" as the one thing that differs between
# them -- rather than comparing an enforced condition to an unenforced one.

NUM_ROUNDS = 3
ATTACK_BUDGET = 3
TARGET_STANCE = -0.8
PIN_FIRST_ROUND = True
STANCE_SAMPLES = 2


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"completed": {}}


def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def record_completion(manifest: dict, key: str, entry: dict):
    """Thread-safe: re-reads the manifest from disk under the lock before
    writing, so a slow trial finishing late doesn't clobber entries another
    concurrent trial already saved in the meantime."""
    with _manifest_lock:
        current = load_manifest()
        current["completed"][key] = entry
        save_manifest(current)
        manifest["completed"] = current["completed"]


def key_for(condition: str, topic_id: str, seed: int) -> str:
    return f"{condition}|{topic_id}|{seed}"


def estimate(topics, seeds, mechanisms, n_agents=8):
    per_trial = (n_agents * STANCE_SAMPLES) * 2 + (n_agents * NUM_ROUNDS)
    n_organic = len(topics) * len(seeds)
    n_adv = len(topics) * len(seeds) * len(mechanisms)
    total_trials = n_organic + n_adv
    total_calls = total_trials * per_trial
    print(f"  Calls per trial: {per_trial}")
    print(f"  Organic control trials: {n_organic}")
    print(f"  Adversarial trials:     {n_adv}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total API calls: ~{total_calls} (excluding retries)")
    return total_calls


def run_seed_cell(topic_id, topic, seed, mechanisms, agents, attacker_id,
                   disinfo_lib, manifest):
    """Everything for one (topic, seed): the organic control, then one
    adversarial trial per mechanism from its frozen pre-stances. Safe to run
    concurrently with other seed-cells -- see module docstring."""
    org_key = key_for("organic", topic_id, seed)
    existing = manifest["completed"].get(org_key)

    if existing is None:
        print(f"\n--- ORGANIC control | {topic_id} | seed={seed} ---")
        org_trial = run_organic_trial(
            LLMClient(provider="nvidia"), agents, topic, num_rounds=NUM_ROUNDS, seed=seed,
            pin_first_round=PIN_FIRST_ROUND,
        )
        path = save_trial(org_trial)
        frozen = dict(org_trial.pre_stances)
        record_completion(manifest, org_key, {
            "trial_id": org_trial.trial_id, "path": path, "pre_stances": frozen,
        })
        print(f"  saved -> {path}")
    else:
        frozen = existing["pre_stances"]
        print(f"\n--- ORGANIC control | {topic_id} | seed={seed} -- already done ---")

    for mechanism in mechanisms:
        adv_key = key_for(mechanism.value, topic_id, seed)
        if adv_key in manifest["completed"]:
            print(f"--- {mechanism.value} | {topic_id} | seed={seed} -- already done, skipping ---")
            continue

        print(f"\n--- {mechanism.value.upper()} | {topic_id} | seed={seed} ---")
        trial, attack_log = run_adversarial_trial(
            LLMClient(provider="nvidia"), agents, topic,
            mechanism=mechanism,
            targeting=Targeting.RANDOM,
            adaptivity=Adaptivity.STATIC,
            target_stance=TARGET_STANCE,
            attack_budget=ATTACK_BUDGET,
            num_rounds=NUM_ROUNDS,
            seed=seed,
            frozen_initial_stances=frozen,
            pin_first_round=PIN_FIRST_ROUND,
            disinfo_lib=disinfo_lib,
            enforce_compliance=(mechanism in ENFORCE_COMPLIANCE_MECHANISMS),
        )
        trial_path, attack_path = save_adversarial_trial(trial, attack_log)
        summarize_adversarial_trial(trial, attack_log, attacker_id)
        record_completion(manifest, adv_key, {
            "trial_id": trial.trial_id, "path": trial_path, "attack_path": attack_path,
        })
        print(f"  saved -> {trial_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--mechanisms", nargs="+", default=DEFAULT_MECHANISMS)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--claims", default="disinfo_claims_template.json")
    parser.add_argument("--parallel", type=int, default=1,
                        help="number of (topic, seed) cells to run concurrently")
    args = parser.parse_args()

    mechanisms = [Mechanism(m) for m in args.mechanisms]

    print("=" * 70)
    print("STEP 6 -- MAIN RUN")
    print("=" * 70)
    estimate(args.topics, args.seeds, mechanisms)
    if args.estimate:
        return

    disinfo_lib = load_disinfo_library(args.claims)
    claim_dependent = {Mechanism.DISINFORMATION, Mechanism.HEDGED_DISINFORMATION}
    if claim_dependent & set(mechanisms):
        for topic_id in args.topics:
            pool = disinfo_lib.get(topic_id, [])
            if not pool:
                print(f"\nERROR: no claims for topic '{topic_id}' in {args.claims}.", file=sys.stderr)
                sys.exit(1)
            for c in pool:
                if "PLACEHOLDER" in c.false_claim:
                    print(f"\nERROR: topic '{topic_id}' still has a PLACEHOLDER claim.", file=sys.stderr)
                    sys.exit(1)
            aligned = [c for c in pool if (c.target_direction >= 0) == (TARGET_STANCE >= 0)]
            if not aligned:
                print(f"\nERROR: no claim in '{topic_id}' aligned with TARGET_STANCE="
                      f"{TARGET_STANCE:+.2f}.", file=sys.stderr)
                sys.exit(1)

    agents = load_agents()
    attacker_id = find_attacker_id(agents)
    manifest = load_manifest()
    print(f"\nAgents: {len(agents)} (attacker={attacker_id})")
    print(f"Already completed (will skip): {len(manifest['completed'])} trials")
    print(f"Parallelism: {args.parallel} concurrent seed-cell(s)\n")

    started = time.time()
    cells = [(topic_id, seed) for topic_id in args.topics for seed in args.seeds]
    topics_cache = {t: load_topic(t) for t in args.topics}

    if args.parallel <= 1:
        for topic_id, seed in cells:
            run_seed_cell(topic_id, topics_cache[topic_id], seed, mechanisms,
                          agents, attacker_id, disinfo_lib, manifest)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(run_seed_cell, topic_id, topics_cache[topic_id], seed,
                         mechanisms, agents, attacker_id, disinfo_lib, manifest): (topic_id, seed)
                for topic_id, seed in cells
            }
            for future in as_completed(futures):
                topic_id, seed = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n!!! CELL FAILED: topic={topic_id} seed={seed}: {e}", file=sys.stderr)

    elapsed = (time.time() - started) / 60
    manifest = load_manifest()
    print(f"\n{'='*70}")
    print(f"RUN COMPLETE -- {len(manifest['completed'])} trials in manifest, "
          f"{elapsed:.1f} min this session")
    print(f"Next: python analyze_step6.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()