import json
import os
import random
import uuid

from src.llm_client import LLMClient
from src.schemas import Agent, TrialRecord
from src.stance import elicit_stance, get_initial_stance
from src.parallel_stance import get_initial_stances_parallel, elicit_stances_parallel
from src.adversary import (
    AttackConfig,
    AttackerController,
    Mechanism,
    Targeting,
    Adaptivity,
    DisinfoClaim,
    run_adversarial_round,
)


def find_attacker_id(agents: list) -> str:
    adversaries = [a.agent_id for a in agents if a.is_adversary]
    if len(adversaries) != 1:
        raise ValueError(f"Expected exactly one is_adversary=true agent, found: {adversaries}")
    return adversaries[0]


def find_high_status_id(agents: list):
    high_status = [a.agent_id for a in agents if a.is_high_status]
    return high_status[0] if high_status else None


def run_adversarial_trial(
    client: LLMClient,
    agents: list,
    topic: dict,
    mechanism: Mechanism,
    targeting: Targeting = Targeting.RANDOM,
    adaptivity: Adaptivity = Adaptivity.STATIC,
    target_stance: float = 0.8,
    attack_budget: int = 3,
    num_rounds: int = 3,
    seed: int = None,
    frozen_initial_stances: dict = None,
    pin_first_round: bool = True,
    disinfo_lib: dict = None,
    enforce_compliance: bool = False,
) -> tuple:
    # Per-trial RNG instance -- see the identical comment in organic_pipeline.
    # AttackerController keeps its OWN separate random.Random(config.seed)
    # instance for target/claim selection, so that part was already isolated;
    # this instance only controls turn-order shuffling within a round.
    rng = random.Random(seed) if seed is not None else random.Random()

    attacker_id = find_attacker_id(agents)
    high_status_id = find_high_status_id(agents)

    model_info = client.info()
    trial = TrialRecord(
        trial_id=str(uuid.uuid4()),
        condition=mechanism.value,
        topic_id=topic["topic_id"],
        seed=seed if seed is not None else -1,
        model_provider=model_info["provider"],
        model_name=model_info["model"],
        temperature=0.3,
        frozen_initial_stances=frozen_initial_stances,
    )

    config = AttackConfig(
        mechanism=mechanism,
        targeting=targeting,
        adaptivity=adaptivity,
        target_stance=target_stance,
        attack_budget=attack_budget,
        attacker_id=attacker_id,
        high_status_agent_id=high_status_id,
        topic=topic["topic_id"],
        seed=seed if seed is not None else -1,
        enforce_compliance=enforce_compliance,
    )
    controller = AttackerController(config, agents, trial_id=trial.trial_id)
    controller.plan_rounds(num_rounds)

    print(f"  [trial {trial.trial_id[:8]}] {mechanism.value} | eliciting pre-stances (parallel)...")
    trial.pre_stances = get_initial_stances_parallel(
        client, agents, topic["prompt_context"], topic["stance_question"],
        frozen_initial_stances,
    )

    stances = dict(trial.pre_stances)

    all_messages = []
    for round_num in range(1, num_rounds + 1):
        print(f"  [trial {trial.trial_id[:8]}] running round {round_num}/{num_rounds}...")
        new_messages = run_adversarial_round(
            client, agents, topic["prompt_context"], all_messages,
            round_number=round_num,
            controller=controller,
            stances=stances,
            total_rounds=num_rounds,
            disinfo_lib=disinfo_lib,
            sim_time_start=(round_num - 1) * 100,
            pin_first_round=pin_first_round,
            rng=rng,
        )
        all_messages.extend(new_messages)
    trial.messages = all_messages

    print(f"  [trial {trial.trial_id[:8]}] eliciting post-stances (parallel)...")
    trial.post_stances = elicit_stances_parallel(
        client, agents, topic["prompt_context"], topic["stance_question"], all_messages,
    )

    return trial, controller.log


def save_adversarial_trial(trial: TrialRecord, attack_log, log_dir: str = "logs") -> tuple:
    os.makedirs(log_dir, exist_ok=True)
    trial_path = os.path.join(
        log_dir,
        f"{trial.condition}_{trial.topic_id}_{trial.model_provider}_{trial.trial_id}.json",
    )
    with open(trial_path, "w") as f:
        json.dump(trial.to_dict(), f, indent=2)

    attack_path = os.path.join(log_dir, f"{trial.trial_id}_attack.json")
    with open(attack_path, "w") as f:
        json.dump(attack_log.to_dict(), f, indent=2)

    return trial_path, attack_path


def summarize_adversarial_trial(trial: TrialRecord, attack_log, attacker_id: str = None):
    print(f"\n{'='*70}")
    print(f"TRIAL SUMMARY: {trial.trial_id[:8]} | condition={trial.condition} | "
          f"topic={trial.topic_id} | seed={trial.seed}")
    print(f"{'='*70}")

    attacker_id = attacker_id or attack_log.attacker_id
    group_shifts = []
    for agent_id in trial.pre_stances:
        pre = trial.pre_stances[agent_id]
        post = trial.post_stances.get(agent_id)
        if post is None:
            continue
        shift = post - pre
        tag = " (attacker)" if agent_id == attacker_id else ""
        print(f"  {agent_id:14s} pre={pre:+.2f}  post={post:+.2f}  shift={shift:+.2f}{tag}")
        if agent_id != attacker_id:
            group_shifts.append(shift)

    avg = sum(group_shifts) / len(group_shifts) if group_shifts else 0.0
    checked = [iv.compliance_signal for iv in attack_log.interventions
               if iv.compliance_signal is not None]
    print(f"\n  Group mean shift (excl. attacker): {avg:+.3f}")
    print(f"  Interventions: {len(attack_log.interventions)} "
          f"(targets: {', '.join(attack_log.target_agents)})")
    if checked:
        print(f"  Compliance: {sum(checked)}/{len(checked)} ({sum(checked)/len(checked):.0%})")
    return avg