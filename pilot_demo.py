"""
Pilot demo -- exercises the REAL integration point (run_adversarial_round)
using the mock LLM client, but loads agents/topics from your REAL
config/personas.json and config/topics.json so agent IDs, personas, and
high-status/adversary flags all match what your actual pipeline uses.

Swap MockLLMClient for your real src.llm_client.LLMClient(provider="nvidia")
to run for real (see main()).

Run from the project root:  python pilot_demo.py
"""
from __future__ import annotations

import json

from src.schemas import Agent
from mock_llm_client import MockLLMClient
from src.adversary import (
    AttackConfig,
    Adaptivity,
    AttackerController,
    Mechanism,
    Targeting,
    load_disinfo_library,
    run_adversarial_round,
)

SEEDS = [1, 2, 3, 4, 5]
MECHANISMS = [
    Mechanism.DISINFORMATION,
    Mechanism.MANUFACTURED_CONSENSUS,
    Mechanism.PERSONALIZED_PERSUASION,
]
PILOT_TOPIC_IDS = ["ai_regulation", "centralized_governance"]  # subset for the pilot
TOTAL_ROUNDS = 3   # matches your established organic_pipeline.py convention (num_rounds=3)
ATTACK_BUDGET = 3
PIN_FIRST_ROUND = True  # must be applied identically to the organic control -- see adversary.py

# Flip to False only once the smoke test below looks right. Full pilot =
# len(MECHANISMS) x len(PILOT_TOPIC_IDS) x len(SEEDS) trials x TOTAL_ROUNDS
# rounds x 8 agents real API calls -- at your 2.5s proactive throttle alone
# that's tens of minutes before any retries. Don't run it blind.
SMOKE_TEST = True
if SMOKE_TEST:
    SEEDS = [1]
    MECHANISMS = [Mechanism.DISINFORMATION]
    PILOT_TOPIC_IDS = ["ai_regulation"]


def load_agents(path: str = "config/personas.json"):
    with open(path) as f:
        raw = json.load(f)
    return [Agent(**a) for a in raw["agents"]]


def load_topics(path: str = "config/topics.json"):
    with open(path) as f:
        raw = json.load(f)
    return {t["topic_id"]: t for t in raw["topics"]}


def find_attacker_id(agents):
    adversaries = [a.agent_id for a in agents if a.is_adversary]
    if len(adversaries) != 1:
        raise ValueError(f"Expected exactly one is_adversary=true agent, found: {adversaries}")
    return adversaries[0]


def find_high_status_id(agents):
    high_status = [a.agent_id for a in agents if a.is_high_status]
    return high_status[0] if high_status else None


def build_stances(agents, attacker_id):
    # Stand-in for TrialRecord.pre_stances -- normally elicited via stance.py
    return {a.agent_id: 0.0 for a in agents if a.agent_id != attacker_id}


def run_one_trial(client, mechanism, topic_id, topic_context, seed, disinfo_lib,
                   attacker_id, high_status_id):
    agents = load_agents()
    stances = build_stances(agents, attacker_id)

    config = AttackConfig(
        mechanism=mechanism,
        targeting=Targeting.RANDOM,
        adaptivity=Adaptivity.STATIC,
        target_stance=0.8,
        attack_budget=ATTACK_BUDGET,
        attacker_id=attacker_id,
        topic=topic_id,
        seed=seed,
    )
    controller = AttackerController(config, agents, trial_id=f"{mechanism.value}_{topic_id}_{seed}")
    controller.plan_rounds(TOTAL_ROUNDS)

    history = []
    for round_number in range(1, TOTAL_ROUNDS + 1):
        new_messages = run_adversarial_round(
            client, agents, topic_context, history, round_number,
            controller, stances, TOTAL_ROUNDS, disinfo_lib=disinfo_lib,
            sim_time_start=(round_number - 1) * len(agents),
            pin_first_round=PIN_FIRST_ROUND,
        )
        history.extend(new_messages)

    return controller.log.to_dict(), history


def main():
    from src.llm_client import LLMClient
    client = LLMClient(provider="nvidia")

    agents = load_agents()
    topics = load_topics()
    attacker_id = find_attacker_id(agents)
    high_status_id = find_high_status_id(agents)
    print(f"Loaded {len(agents)} agents from config/personas.json "
          f"(attacker={attacker_id}, high_status={high_status_id})")

    n_trials = len(MECHANISMS) * len(PILOT_TOPIC_IDS) * len(SEEDS)
    n_calls = n_trials * TOTAL_ROUNDS * len(agents)
    print(f"{'SMOKE TEST -- ' if SMOKE_TEST else ''}"
          f"{n_trials} trials, ~{n_calls} real API calls "
          f"({len(MECHANISMS)} mechanisms x {len(PILOT_TOPIC_IDS)} topics x {len(SEEDS)} seeds)\n")

    disinfo_lib = load_disinfo_library("disinfo_claims_template.json")

    all_logs = []
    sample_history = None
    trial_num = 0
    for mechanism in MECHANISMS:
        for topic_id in PILOT_TOPIC_IDS:
            topic_context = topics[topic_id]["prompt_context"]
            for seed in SEEDS:
                trial_num += 1
                print(f"[{trial_num}/{n_trials}] running {mechanism.value} / {topic_id} / seed={seed} ...")
                log, history = run_one_trial(
                    client, mechanism, topic_id, topic_context, seed, disinfo_lib,
                    attacker_id, high_status_id,
                )
                all_logs.append(log)
                if mechanism == Mechanism.DISINFORMATION and sample_history is None:
                    sample_history = history

    print(f"\nRan {len(all_logs)} pilot trials through run_adversarial_round "
          f"({len(MECHANISMS)} mechanisms x {len(PILOT_TOPIC_IDS)} topics x {len(SEEDS)} seeds).")

    counts = [len(t["interventions"]) for t in all_logs]
    print(f"Interventions per trial -- min={min(counts)}, max={max(counts)}, expected={ATTACK_BUDGET}")

    compliance_checked = [iv["compliance_signal"] for t in all_logs for iv in t["interventions"]
                           if iv["compliance_signal"] is not None]
    if compliance_checked:
        rate = sum(compliance_checked) / len(compliance_checked)
        print(f"Compliance rate (message plausibly matches its mechanism): "
              f"{sum(compliance_checked)}/{len(compliance_checked)} ({rate:.0%})")

    print("\nSample full round transcript (first disinformation trial):")
    for m in sample_history:
        tag = f"{m.agent_id}*" if m.agent_id == attacker_id else m.agent_id
        target = f" -> {m.target_agent_id}" if m.target_agent_id else ""
        print(f"  [round {m.round_number}] {tag}{target}: {m.content}")

    print("\nSample attack log:")
    print(json.dumps(next(t for t in all_logs if t["condition"] == "disinformation"), indent=2))


if __name__ == "__main__":
    main()