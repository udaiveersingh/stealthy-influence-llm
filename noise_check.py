"""
Measurement-noise check: elicit each agent's stance TWICE in a row with
NOTHING happening in between (same empty conversation history both times).

If the two numbers differ meaningfully, that's pure elicitation noise --
not persuasion -- and we need to fix this before trusting any stance-shift
result from the real pipeline.

Usage:
    python noise_check.py
"""
import json
from src.llm_client import LLMClient
from src.schemas import Agent
from src.stance import elicit_stance


def load_agents(path="config/personas.json") -> list[Agent]:
    with open(path) as f:
        data = json.load(f)
    return [Agent(**a) for a in data["agents"] if not a["is_adversary"]]


def load_topic(topic_id: str, path="config/topics.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    for t in data["topics"]:
        if t["topic_id"] == topic_id:
            return t
    raise ValueError(f"Topic {topic_id} not found")


def main():
    client = LLMClient()
    agents = load_agents()
    topic = load_topic("remote_work")

    print("Eliciting stance TWICE per agent, same empty history both times.")
    print("(Large differences here = pure noise, not persuasion.)\n")

    diffs = []
    for agent in agents:
        v1, _ = elicit_stance(client, agent, topic["prompt_context"], topic["stance_question"], [])
        v2, _ = elicit_stance(client, agent, topic["prompt_context"], topic["stance_question"], [])
        diff = abs(v2 - v1)
        diffs.append(diff)
        flag = "  <-- HIGH NOISE" if diff >= 0.3 else ""
        print(f"  {agent.agent_id:12s} run1={v1:+.2f}  run2={v2:+.2f}  |diff|={diff:.2f}{flag}")

    avg_diff = sum(diffs) / len(diffs)
    print(f"\nAverage |diff| across agents: {avg_diff:.3f}")
    print("For reference: your real stance-shift effects need to be reliably")
    print("larger than this noise floor, or they're not distinguishable from noise.")


if __name__ == "__main__":
    main()