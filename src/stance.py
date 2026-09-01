"""
Stance elicitation: ask an agent (in persona) for its current numeric stance
on a topic, given whatever conversation history it has seen so far.
"""
import re
from src.llm_client import LLMClient
from src.schemas import Agent, Message


def _format_history(messages: list[Message], max_messages: int = 10) -> str:
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
    num_samples: int = 2,
) -> tuple[float, str]:
    """
    Returns (stance_value, raw_model_response_from_first_sample).

    To reduce measurement noise, this draws `num_samples` independent stance
    readings at low temperature and returns their mean. The raw text of the
    first sample is returned alongside for logging/debugging purposes only --
    the numeric value is what should be used downstream.

    num_samples defaults to 2 (down from 3) to reduce token usage for
    volume runs -- noise floor was ~0.05 at 3 samples; re-run noise_check.py
    if you need to confirm 2 samples is still acceptable.
    """
    system_prompt = (
        f"{agent.persona_description}\n\n"
        "You are participating in a group discussion. Stay fully in character. "
        "Always follow the requested response format exactly. Be extremely brief."
    )
    history_text = _format_history(conversation_history)
    user_prompt = (
        f"{topic_context}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"{stance_question} Keep your explanation to under 15 words."
    )

    samples = []
    first_raw = None
    for i in range(num_samples):
        last_error = None
        for attempt in range(max_retries + 1):
            # Low temperature: this is a measurement instrument, not creative
            # writing -- we want consistency, not variety, from each single draw.
            raw = client.complete(system_prompt, user_prompt, temperature=0.3, max_tokens=150)
            try:
                value = _parse_stance_number(raw)
                samples.append(value)
                if first_raw is None:
                    first_raw = raw
                break
            except ValueError as e:
                last_error = e
                continue
        else:
            raise RuntimeError(
                f"Failed to parse a stance number for agent {agent.agent_id} "
                f"after {max_retries + 1} attempts on sample {i+1}. Last error: {last_error}"
            )

    mean_value = sum(samples) / len(samples)
    return mean_value, first_raw