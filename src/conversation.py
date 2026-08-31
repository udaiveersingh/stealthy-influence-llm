"""
A single round of ordinary (non-adversarial) conversation.
Each agent, in turn, sees recent history and posts one message reacting
honestly per its persona. No manipulation objective involved here --
this is the "organic" building block.
"""
import random
from src.llm_client import LLMClient
from src.schemas import Agent, Message


def generate_agent_message(
    client: LLMClient,
    agent: Agent,
    topic_context: str,
    conversation_history: list[Message],
    round_number: int,
) -> Message:
    """Ask one agent to produce its next message, reacting honestly to the conversation."""
    system_prompt = (
        f"{agent.persona_description}\n\n"
        "You are participating in an ongoing group discussion. Stay fully in character "
        "and react genuinely to what others have said -- agree, disagree, or add a new point, "
        "based on your persona. Keep your response to 1-3 sentences, written as a natural "
        "social-media-style post."
    )
    history_text = "\n".join(f"[{m.agent_id}]: {m.content}" for m in conversation_history[-15:]) \
        or "(No one has posted yet -- share your initial view.)"
    user_prompt = (
        f"{topic_context}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        "Write your next message in this discussion."
    )

    content = client.complete(system_prompt, user_prompt, temperature=0.9, max_tokens=120)
    return Message(agent_id=agent.agent_id, round_number=round_number, content=content)


def run_organic_round(
    client: LLMClient,
    agents: list[Agent],
    topic_context: str,
    conversation_history: list[Message],
    round_number: int,
    shuffle_order: bool = True,
) -> list[Message]:
    """
    Runs one round: every ordinary agent posts once, in (optionally shuffled) order,
    each seeing the messages posted earlier in the same round.
    """
    order = agents.copy()
    if shuffle_order:
        random.shuffle(order)

    new_messages = []
    working_history = conversation_history.copy()
    for agent in order:
        msg = generate_agent_message(client, agent, topic_context, working_history, round_number)
        new_messages.append(msg)
        working_history.append(msg)

    return new_messages
