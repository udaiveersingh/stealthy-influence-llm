"""
Full organic baseline pipeline.

FIXES applied in this version:
1. 8-vs-8 population parity: load_agents() now INCLUDES the adversary agent
   by default (it just behaves as an ordinary agent in the organic condition).
   Previously it was excluded, giving organic trials 7 agents vs. 8 in future
   adversarial trials -- a real confound. Pass exclude_adversary=True only if
   you specifically need the old 7-agent behavior for some analysis.
2. Frozen initial-stance support: pass frozen_initial_stances={agent_id: value}
   to skip fresh pre-elicitation and start every agent from a fixed state --
   needed for clean paired organic/adversarial comparisons later.
3. Every TrialRecord now stamps model_provider/model_name/temperature, so
   gpt-oss vs. Nemotron data can never be silently mixed up during analysis.
4. Uses synthetic sim_time (round/slot index) instead of relying solely on
   wall-clock timestamps for message ordering.
5. pin_first_round threaded through to run_organic_round. MUST be set the
   same way here as in the adversarial condition's run_adversarial_round --
   otherwise the two conditions get different amounts of context, which is
   a confound independent of anything the attacker does.
"""
import json
import os
import random
import uuid

from src.llm_client import LLMClient
from src.schemas import Agent, TrialRecord
from src.stance import elicit_stance, get_initial_stance
from src.parallel_stance import get_initial_stances_parallel, elicit_stances_parallel
from src.conversation import run_organic_round


def load_agents(path="config/personas.json", exclude_adversary=False) -> list:
    with open(path) as f:
        data = json.load(f)
    agents = [Agent(**a) for a in data["agents"]]
    if exclude_adversary:
        agents = [a for a in agents if not a.is_adversary]
    return agents


def load_topic(topic_id: str, path="config/topics.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    for t in data["topics"]:
        if t["topic_id"] == topic_id:
            return t
    raise ValueError(f"Topic {topic_id} not found")


def run_organic_trial(
    client: LLMClient,
    agents: list,
    topic: dict,
    num_rounds: int = 3,
    seed: int = None,
    frozen_initial_stances: dict = None,
    pin_first_round: bool = False,
) -> TrialRecord:
    if seed is not None:
        random.seed(seed)

    model_info = client.info()
    trial = TrialRecord(
        trial_id=str(uuid.uuid4()),
        condition="organic",
        topic_id=topic["topic_id"],
        seed=seed if seed is not None else -1,
        model_provider=model_info["provider"],
        model_name=model_info["model"],
        temperature=0.3,  # stance elicitation temperature; conversation uses 0.9 internally
        frozen_initial_stances=frozen_initial_stances,
    )

    print(f"  [trial {trial.trial_id[:8]}] eliciting pre-stances (parallel)...")
    trial.pre_stances = get_initial_stances_parallel(
        client, agents, topic["prompt_context"], topic["stance_question"],
        frozen_initial_stances,
    )

    all_messages = []
    for round_num in range(1, num_rounds + 1):
        print(f"  [trial {trial.trial_id[:8]}] running round {round_num}/{num_rounds}...")
        new_messages = run_organic_round(
            client, agents, topic["prompt_context"], all_messages,
            round_number=round_num, sim_time_start=(round_num - 1) * 100,
            pin_first_round=pin_first_round,
        )
        all_messages.extend(new_messages)
    trial.messages = all_messages

    print(f"  [trial {trial.trial_id[:8]}] eliciting post-stances (parallel)...")
    trial.post_stances = elicit_stances_parallel(
        client, agents, topic["prompt_context"], topic["stance_question"], all_messages,
    )

    return trial


def save_trial(trial: TrialRecord, log_dir: str = "logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(
        log_dir,
        f"{trial.condition}_{trial.topic_id}_{trial.model_provider}_{trial.trial_id}.json",
    )
    with open(path, "w") as f:
        json.dump(trial.to_dict(), f, indent=2)
    return path


def summarize_trial(trial: TrialRecord):
    print(f"\n{'='*70}")
    print(f"TRIAL SUMMARY: {trial.trial_id[:8]} | condition={trial.condition} | "
          f"topic={trial.topic_id} | model={trial.model_provider}/{trial.model_name}")
    print(f"{'='*70}")
    shifts = []
    for agent_id in trial.pre_stances:
        pre = trial.pre_stances[agent_id]
        post = trial.post_stances.get(agent_id)
        if post is None:
            continue
        shift = post - pre
        shifts.append(shift)
        print(f"  {agent_id:12s} pre={pre:+.2f}  post={post:+.2f}  shift={shift:+.2f}")
    avg_shift = sum(shifts) / len(shifts) if shifts else 0.0
    print(f"\n  Average stance shift: {avg_shift:+.3f}")
    print(f"  Total messages generated: {len(trial.messages)}")
    print(f"  Agent count: {len(trial.pre_stances)}  (should be 8 for parity with future adversarial trials)")
    return avg_shift


if __name__ == "__main__":
    client = LLMClient()
    agents = load_agents()  # now includes all 8 by default
    topic = load_topic("remote_work")

    # pin_first_round=True to match pilot_demo.py's adversarial trials --
    # change both together if you ever change one.
    trial = run_organic_trial(client, agents, topic, num_rounds=3, seed=42, pin_first_round=True)
    avg_shift = summarize_trial(trial)
    path = save_trial(trial)
    print(f"\nSaved trial to: {path}")