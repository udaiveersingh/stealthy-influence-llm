"""
Sanity-check script for Steps 1-3.
Run this after setting GROQ_API_KEY to confirm:
  1. Stance elicitation returns parseable numbers consistently.
  2. A single organic conversation round produces sensible, in-persona messages.

Usage:
    export GROQ_API_KEY=your_key_here
    python3 test_skeleton.py
"""
import json
import sys
from src.llm_client import LLMClient
from src.schemas import Agent
from src.stance import elicit_stance
from src.conversation import run_organic_round


def load_agents(path="config/personas.json") -> list[Agent]:
    with open(path) as f:
        data = json.load(f)
    return [Agent(**a) for a in data["agents"]]


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
    ordinary_agents = [a for a in agents if not a.is_adversary]  # exclude adversary for now
    topic = load_topic("remote_work")

    print("=" * 70)
    print("STEP A: Eliciting PRE-conversation stances")
    print("=" * 70)
    pre_stances = {}
    for agent in ordinary_agents:
        value, raw = elicit_stance(
            client, agent, topic["prompt_context"], topic["stance_question"], []
        )
        pre_stances[agent.agent_id] = value
        print(f"  {agent.agent_id:12s} stance={value:+.2f}   raw='{raw}'")

    print("\n" + "=" * 70)
    print("STEP B: Running one organic conversation round")
    print("=" * 70)
    messages = run_organic_round(client, ordinary_agents, topic["prompt_context"], [], round_number=1)
    for m in messages:
        print(f"  [{m.agent_id}]: {m.content}")

    print("\n" + "=" * 70)
    print("STEP C: Eliciting POST-round stances (given the round's messages)")
    print("=" * 70)
    post_stances = {}
    for agent in ordinary_agents:
        value, raw = elicit_stance(
            client, agent, topic["prompt_context"], topic["stance_question"], messages
        )
        post_stances[agent.agent_id] = value
        shift = value - pre_stances[agent.agent_id]
        print(f"  {agent.agent_id:12s} stance={value:+.2f}   shift={shift:+.2f}")

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    avg_shift = sum(post_stances[a.agent_id] - pre_stances[a.agent_id] for a in ordinary_agents) / len(ordinary_agents)
    print(f"  Average stance shift after 1 round: {avg_shift:+.3f}")
    print(f"  (Sanity: is this near zero after just 1 round? That's expected -- ")
    print(f"   real shift accumulates over multiple rounds. Main thing to check: ")
    print(f"   did every stance parse successfully, and do the posted messages ")
    print(f"   actually sound like their personas?)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
