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
    # Per-trial RNG instance instead of global random.seed(). The global
    # module is shared process-wide, so seeding it from a trial running
    # concurrently with other trials (different seeds) would race and make
    # every trial's shuffle order depend on scheduling, breaking
    # reproducibility. A local instance is fully isolated per trial and safe
    # to run from multiple threads at once.
    rng = random.Random(seed) if seed is not None else random.Random()

    model_info = client.info()
    trial = TrialRecord(
        trial_id=str(uuid.uuid4()),
        condition="organic",
        topic_id=topic["topic_id"],
        seed=seed if seed is not None else -1,
        model_provider=model_info["provider"],
        model_name=model_info["model"],
        temperature=0.3,
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
            pin_first_round=pin_first_round, rng=rng,
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
    agents = load_agents()
    topic = load_topic("remote_work")

    trial = run_organic_trial(client, agents, topic, num_rounds=3, seed=42, pin_first_round=True)
    avg_shift = summarize_trial(trial)
    path = save_trial(trial)
    print(f"\nSaved trial to: {path}")