"""
Stance elicitation: ask an agent (in persona) for its current numeric stance
on a topic, given whatever conversation history it has seen so far.
"""
import re
from src.llm_client import LLMClient
from src.schemas import Agent, Message


def _format_history(messages: list[Message], max_messages: int = 20) -> str:
    """Render recent conversation history as plain text for the prompt."""
    if not messages:
        return "(No conversation has happened yet.)"
    recent = messages[-max_messages:]
    lines = [f"[{m.agent_id}]: {m.content}" for m in recent]
    return "\n".join(lines)


def _parse_stance_number(raw_text: str) -> float:
    """
    Extract the first float-looking number in [-1, 1] from the model's response.
    Raises ValueError if nothing parseable is found -- caller should retry.
    """
    match = re.search(r"-?\d*\.?\d+", raw_text)
    if not match:
        raise ValueError(f"No number found in response: {raw_text!r}")
    value = float(match.group())
    # Clamp defensively -- models occasionally drift outside the requested range.
    value = max(-1.0, min(1.0, value))
    return value


def elicit_stance(
    client: LLMClient,
    agent: Agent,
    topic_context: str,
    stance_question: str,
    conversation_history: list[Message],
    max_retries: int = 2,
) -> tuple[float, str]:
    """
    Returns (stance_value, raw_model_response).
    Retries a small number of times if parsing fails.
    """
    system_prompt = (
        f"{agent.persona_description}\n\n"
        "You are participating in a group discussion. Stay fully in character. "
        "Always follow the requested response format exactly."
    )
    history_text = _format_history(conversation_history)
    user_prompt = (
        f"{topic_context}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"{stance_question}"
    )

    last_error = None
    for attempt in range(max_retries + 1):
        raw = client.complete(system_prompt, user_prompt, temperature=0.7, max_tokens=100)
        try:
            value = _parse_stance_number(raw)
            return value, raw
        except ValueError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to parse a stance number for agent {agent.agent_id} "
        f"after {max_retries + 1} attempts. Last error: {last_error}"
    )
