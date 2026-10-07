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
from src.conversation import run_organic_round, _first_round_messages


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
    num_attack_rounds: int = 3,
    num_removal_rounds: int = 0,
    seed: int = None,
    frozen_initial_stances: dict = None,
    pin_first_round: bool = True,
    disinfo_lib: dict = None,
    enforce_compliance: bool = False,
) -> tuple:
    """num_removal_rounds=0 (default) reproduces the exact prior behavior:
    attack rounds only, trial.post_stances is the final stance, no persistence
    data. num_removal_rounds>0 adds a phase AFTER the attack where the
    attacker is not special-cased at all -- it runs through the ordinary
    run_organic_round() path, so it has no hidden objective, no budget, no
    memory of its own prior interventions fed back in, nothing. It is
    genuinely just another agent for these rounds, not an attacker that
    happens to have run out of budget.

    trial.post_stances = stance at the END OF THE ATTACK PHASE (same meaning
    as before -- unchanged field, so old analysis code keeps working).
    trial.post_removal_stances = stance after the removal phase, ONLY
    populated when num_removal_rounds>0.
    """
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
        topic_context=topic["prompt_context"],
        scale_description=topic["stance_question"],
        enforce_compliance=enforce_compliance,
    )
    controller = AttackerController(config, agents, trial_id=trial.trial_id)
    controller.plan_rounds(num_attack_rounds)

    print(f"  [trial {trial.trial_id[:8]}] {mechanism.value} | eliciting pre-stances (parallel)...")
    trial.pre_stances = get_initial_stances_parallel(
        client, agents, topic["prompt_context"], topic["stance_question"],
        frozen_initial_stances,
    )

    stances = dict(trial.pre_stances)

    all_messages = []
    for round_num in range(1, num_attack_rounds + 1):
        print(f"  [trial {trial.trial_id[:8]}] attack round {round_num}/{num_attack_rounds}...")
        new_messages = run_adversarial_round(
            client, agents, topic["prompt_context"], all_messages,
            round_number=round_num,
            controller=controller,
            stances=stances,
            total_rounds=num_attack_rounds,
            disinfo_lib=disinfo_lib,
            sim_time_start=(round_num - 1) * 100,
            pin_first_round=pin_first_round,
            rng=rng,
        )
        all_messages.extend(new_messages)

    print(f"  [trial {trial.trial_id[:8]}] eliciting end-of-attack stances (parallel)...")
    trial.post_stances = elicit_stances_parallel(
        client, agents, topic["prompt_context"], topic["stance_question"], all_messages,
    )
    trial.messages = list(all_messages)  # attack-phase messages only, until/unless removal runs

    if num_removal_rounds > 0:
        attack_messages_snapshot = list(all_messages)  # pinned into removal-phase stance elicitation below
        for i in range(num_removal_rounds):
            round_num = num_attack_rounds + i + 1
            print(f"  [trial {trial.trial_id[:8]}] removal round {round_num} "
                  f"({i+1}/{num_removal_rounds}, attacker silent)...")
            new_messages = run_organic_round(
                client, agents, topic["prompt_context"], all_messages,
                round_number=round_num,
                sim_time_start=num_attack_rounds * 100 + i * 100,
                pin_first_round=pin_first_round,
                rng=rng,
            )
            all_messages.extend(new_messages)
        trial.messages = all_messages

        print(f"  [trial {trial.trial_id[:8]}] eliciting post-removal stances (parallel, "
              f"pinning attack-phase context)...")
        trial.post_removal_stances = elicit_stances_parallel(
            client, agents, topic["prompt_context"], topic["stance_question"], all_messages,
            pinned_prefix=attack_messages_snapshot,
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
    has_removal = bool(trial.post_removal_stances)
    group_shifts = []
    recoveries = []
    for agent_id in trial.pre_stances:
        pre = trial.pre_stances[agent_id]
        post = trial.post_stances.get(agent_id)
        if post is None:
            continue
        shift = post - pre
        tag = " (attacker)" if agent_id == attacker_id else ""
        line = f"  {agent_id:14s} pre={pre:+.2f}  end_attack={post:+.2f}  shift={shift:+.2f}"
        if has_removal:
            removed = trial.post_removal_stances.get(agent_id)
            if removed is not None:
                recovery = removed - post  # +ve = kept moving same way after attacker left; -ve = reverted
                line += f"  post_removal={removed:+.2f}  recovery={recovery:+.2f}"
                if agent_id != attacker_id:
                    recoveries.append(recovery)
        print(line + tag)
        if agent_id != attacker_id:
            group_shifts.append(shift)

    avg = sum(group_shifts) / len(group_shifts) if group_shifts else 0.0
    checked = [iv.compliance_signal for iv in attack_log.interventions
               if iv.compliance_signal is not None]
    print(f"\n  Group mean shift, end of attack (excl. attacker): {avg:+.3f}")
    if has_removal and recoveries:
        avg_recovery = sum(recoveries) / len(recoveries)
        print(f"  Group mean change during removal (excl. attacker): {avg_recovery:+.3f}")
        if abs(avg) > 1e-6:
            print(f"  Persistence ratio (retained / attack effect): "
                  f"{(avg + avg_recovery) / avg:+.2f}")
    print(f"  Interventions: {len(attack_log.interventions)} "
          f"(targets: {', '.join(attack_log.target_agents)})")
    if checked:
        print(f"  Compliance: {sum(checked)}/{len(checked)} ({sum(checked)/len(checked):.0%})")
    return avg