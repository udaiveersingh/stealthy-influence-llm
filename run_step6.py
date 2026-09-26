"""
Step 6 main data collection.

Design:
  For each (topic, seed):
    1. Run ONE organic control trial (8 agents, A8 behaving ordinarily).
    2. Take its pre_stances as frozen_initial_stances.
    3. Run one adversarial trial per mechanism, all starting from those
       SAME frozen stances.

  That makes every adversarial trial directly comparable to its own organic
  control -- same agents, same topic, same starting stances, same round
  count, same context rules. The only difference is A8's hidden objective.

Resumability: every completed trial writes a marker into runs/manifest.json.
Re-running skips anything already done, so an interrupted run (or a rate
limit, or a crash 90 minutes in) costs you nothing but the trial in flight.

Usage:
    python run_step6.py --estimate          # cost/time estimate, runs nothing
    python run_step6.py                     # run with defaults
    python run_step6.py --topics ai_regulation --seeds 1 2 3
    python run_step6.py --mechanisms disinformation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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

DEFAULT_TOPICS = ["ai_regulation"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_MECHANISMS = ["disinformation", "manufactured_consensus", "personalized_persuasion"]

NUM_ROUNDS = 3
ATTACK_BUDGET = 3
TARGET_STANCE = -0.8
PIN_FIRST_ROUND = True   # must match organic control -- both set here together
STANCE_SAMPLES = 2       # elicit_stance default; used only for cost estimation


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"completed": {}}


def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def key_for(condition: str, topic_id: str, seed: int) -> str:
    return f"{condition}|{topic_id}|{seed}"


def estimate(topics, seeds, mechanisms, n_agents=8):
    per_trial = (n_agents * STANCE_SAMPLES) * 2 + (n_agents * NUM_ROUNDS)
    n_organic = len(topics) * len(seeds)
    n_adv = len(topics) * len(seeds) * len(mechanisms)
    total_trials = n_organic + n_adv
    total_calls = total_trials * per_trial
    print(f"  Calls per trial: {per_trial} "
          f"({n_agents*STANCE_SAMPLES} pre + {n_agents*NUM_ROUNDS} conversation + "
          f"{n_agents*STANCE_SAMPLES} post)")
    print(f"  Organic control trials: {n_organic}")
    print(f"  Adversarial trials:     {n_adv} "
          f"({len(mechanisms)} mechanisms x {len(topics)} topics x {len(seeds)} seeds)")
    print(f"  Total trials: {total_trials}")
    print(f"  Total API calls: ~{total_calls} (excluding retries)")
    for rate in (2.5, 4.0):
        print(f"    at ~{rate}s/call: ~{total_calls*rate/60:.0f} min")
    return total_calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--mechanisms", nargs="+", default=DEFAULT_MECHANISMS)
    parser.add_argument("--estimate", action="store_true", help="print cost estimate and exit")
    parser.add_argument("--claims", default="disinfo_claims_template.json")
    args = parser.parse_args()

    mechanisms = [Mechanism(m) for m in args.mechanisms]

    print("=" * 70)
    print("STEP 6 -- MAIN RUN")
    print("=" * 70)
    estimate(args.topics, args.seeds, mechanisms)
    if args.estimate:
        return

    disinfo_lib = load_disinfo_library(args.claims)
    # Fail loudly BEFORE spending an hour of API calls if a topic has no claim.
    if Mechanism.DISINFORMATION in mechanisms:
        for topic_id in args.topics:
            pool = disinfo_lib.get(topic_id, [])
            if not pool:
                print(f"\nERROR: disinformation mechanism requested but no claims exist for "
                      f"topic '{topic_id}' in {args.claims}. Add claims or drop the topic/"
                      f"mechanism.", file=sys.stderr)
                sys.exit(1)
            for c in pool:
                if "PLACEHOLDER" in c.false_claim:
                    print(f"\nERROR: topic '{topic_id}' still has a PLACEHOLDER claim in "
                          f"{args.claims}. Replace it with a real, human-reviewed claim "
                          f"before collecting data.", file=sys.stderr)
                    sys.exit(1)

    client = LLMClient(provider="nvidia")
    agents = load_agents()  # includes A8_ADVERSARY
    attacker_id = find_attacker_id(agents)
    manifest = load_manifest()
    print(f"\nAgents: {len(agents)} (attacker={attacker_id})")
    print(f"Already completed (will skip): {len(manifest['completed'])} trials\n")

    started = time.time()
    for topic_id in args.topics:
        topic = load_topic(topic_id)

        for seed in args.seeds:
            # ---- 1. Organic control for this (topic, seed) ----
            org_key = key_for("organic", topic_id, seed)
            frozen = manifest["completed"].get(org_key, {}).get("pre_stances")

            if frozen is None:
                print(f"\n--- ORGANIC control | {topic_id} | seed={seed} ---")
                org_trial = run_organic_trial(
                    client, agents, topic, num_rounds=NUM_ROUNDS, seed=seed,
                    pin_first_round=PIN_FIRST_ROUND,
                )
                path = save_trial(org_trial)
                frozen = dict(org_trial.pre_stances)
                manifest["completed"][org_key] = {
                    "trial_id": org_trial.trial_id,
                    "path": path,
                    "pre_stances": frozen,
                }
                save_manifest(manifest)
                print(f"  saved -> {path}")
            else:
                print(f"\n--- ORGANIC control | {topic_id} | seed={seed} -- already done, "
                      f"reusing its pre-stances ---")

            # ---- 2. One adversarial trial per mechanism, same frozen start ----
            for mechanism in mechanisms:
                adv_key = key_for(mechanism.value, topic_id, seed)
                if adv_key in manifest["completed"]:
                    print(f"--- {mechanism.value} | {topic_id} | seed={seed} -- already done, "
                          f"skipping ---")
                    continue

                print(f"\n--- {mechanism.value.upper()} | {topic_id} | seed={seed} ---")
                trial, attack_log = run_adversarial_trial(
                    client, agents, topic,
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
                )
                trial_path, attack_path = save_adversarial_trial(trial, attack_log)
                summarize_adversarial_trial(trial, attack_log, attacker_id)
                manifest["completed"][adv_key] = {
                    "trial_id": trial.trial_id,
                    "path": trial_path,
                    "attack_path": attack_path,
                }
                save_manifest(manifest)
                print(f"  saved -> {trial_path}")

    elapsed = (time.time() - started) / 60
    print(f"\n{'='*70}")
    print(f"RUN COMPLETE -- {len(manifest['completed'])} trials in manifest, "
          f"{elapsed:.1f} min this session")
    print(f"Next: python analyze_step6.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
