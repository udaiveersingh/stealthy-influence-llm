"""
Step 4: Full organic baseline pipeline.

Runs: pre-stance elicitation -> N conversation rounds -> post-stance elicitation,
logs everything into a TrialRecord, and saves it as JSON under logs/.

This is the building block every later condition (disinformation, manufactured
consensus, personalized persuasion) will extend -- get this right and the
adversarial pipelines are mostly just swapping in an adversary's message
generator for one agent.
"""
import json
import os
import random
import time
import uuid

from src.llm_client import LLMClient
from src.schemas import Agent, TrialRecord
from src.stance import elicit_stance
from src.conversation import run_organic_round


def load_agents(path="config/personas.json", exclude_adversary=True) -> list[Agent]:
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
    agents: list[Agent],
    topic: dict,
    num_rounds: int = 3,
    seed: int = None,
) -> TrialRecord:
    """
    Runs one full organic trial: pre-stance -> num_rounds of conversation -> post-stance.
    Returns a populated TrialRecord.
    """
    if seed is not None:
        random.seed(seed)

    trial = TrialRecord(
        trial_id=str(uuid.uuid4()),
        condition="organic",
        topic_id=topic["topic_id"],
        seed=seed if seed is not None else -1,
    )

    # --- Pre-stance ---
    print(f"  [trial {trial.trial_id[:8]}] eliciting pre-stances...")
    for agent in agents:
        value, _ = elicit_stance(client, agent, topic["prompt_context"], topic["stance_question"], [])
        trial.pre_stances[agent.agent_id] = value

    # --- Conversation rounds ---
    all_messages = []
    for round_num in range(1, num_rounds + 1):
        print(f"  [trial {trial.trial_id[:8]}] running round {round_num}/{num_rounds}...")
        new_messages = run_organic_round(
            client, agents, topic["prompt_context"], all_messages, round_number=round_num
        )
        all_messages.extend(new_messages)
    trial.messages = all_messages

    # --- Post-stance ---
    print(f"  [trial {trial.trial_id[:8]}] eliciting post-stances...")
    for agent in agents:
        value, _ = elicit_stance(client, agent, topic["prompt_context"], topic["stance_question"], all_messages)
        trial.post_stances[agent.agent_id] = value

    return trial


def save_trial(trial: TrialRecord, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{trial.condition}_{trial.topic_id}_{trial.trial_id}.json")
    with open(path, "w") as f:
        json.dump(trial.to_dict(), f, indent=2)
    return path


def summarize_trial(trial: TrialRecord):
    print(f"\n{'='*70}")
    print(f"TRIAL SUMMARY: {trial.trial_id[:8]} | condition={trial.condition} | topic={trial.topic_id}")
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
    return avg_shift


if __name__ == "__main__":
    client = LLMClient()
    agents = load_agents()
    topic = load_topic("remote_work")

    trial = run_organic_trial(client, agents, topic, num_rounds=3, seed=42)
    avg_shift = summarize_trial(trial)
    path = save_trial(trial)
    print(f"\nSaved trial to: {path}")
    print("\nIf this looks sane (messages coherent, stances parsed, avg shift not")
    print("wildly implausible), the organic pipeline is validated. Next: adversary logic.")
