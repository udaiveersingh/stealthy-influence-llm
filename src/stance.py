import re
from src.llm_client import LLMClient
from src import labels
from src.schemas import Agent, Message


def _format_history(messages: list, max_messages: int = 10, pinned_prefix: list = None) -> str:
    """pinned_prefix: messages always shown regardless of the rolling window --
    needed for persistence: post-removal stance elicitation happens after the
    attack phase has scrolled well outside a 10-message window (6 rounds x 8
    agents = 48 messages), so without this, an agent asked "has your view
    changed since the attack?" literally cannot see what the attack was."""
    if not messages:
        return "(No conversation has happened yet.)"
    recent = messages[-max_messages:]

    if not pinned_prefix:
        lines = [f"[{labels.display_id(m.agent_id)}]: {m.content}" for m in recent]
        return "\n".join(lines)

    pinned_ids = {id(m) for m in pinned_prefix}
    recent = [m for m in recent if id(m) not in pinned_ids]
    lines = ["[Earlier in this discussion:]"]
    lines += [f"[{labels.display_id(m.agent_id)}]: {m.content}" for m in pinned_prefix]
    if recent:
        lines.append("[More recently:]")
        lines += [f"[{labels.display_id(m.agent_id)}]: {m.content}" for m in recent]
    return "\n".join(lines)


def _parse_stance_number(raw_text: str) -> float:
    marker_matches = re.findall(r"STANCE:\s*(-?\d*\.?\d+)", raw_text, re.IGNORECASE)
    if not marker_matches:
        raise ValueError(
            f"No STANCE: marker found (likely truncated before reaching an answer): {raw_text[-200:]!r}"
        )
    value = float(marker_matches[-1])
    return max(-1.0, min(1.0, value))


def elicit_stance(
    client: LLMClient,
    agent: Agent,
    topic_context: str,
    stance_question: str,
    conversation_history: list,
    max_retries: int = 2,
    num_samples: int = 2,
    pinned_prefix: list = None,
) -> tuple:
    system_prompt = (
        f"{agent.persona_description}\n\n"
        "You are participating in a group discussion. Stay fully in character. "
        "Do not show any reasoning, thinking process, or meta-commentary -- respond "
        "with only the requested final answer, nothing else."
    )
    history_text = _format_history(conversation_history, pinned_prefix=pinned_prefix)
    user_prompt = (
        f"{topic_context}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"{stance_question}\n\n"
        "Respond in EXACTLY this format, with nothing before or after it:\n"
        "STANCE: <number between -1 and 1>\n"
        "REASON: <one short sentence, under 15 words>"
    )

    samples = []
    first_raw = None
    for i in range(num_samples):
        last_error = None
        for attempt in range(max_retries + 1):
            raw = client.complete(system_prompt, user_prompt, temperature=0.3, max_tokens=1200)
            try:
                value = _parse_stance_number(raw)
                samples.append(value)
                if first_raw is None:
                    first_raw = raw
                break
            except ValueError as e:
                last_error = e
                print(f"    [stance retry {attempt+1}/{max_retries+1}] {agent.agent_id}: {e}")
                continue
        else:
            raise RuntimeError(
                f"Failed to parse a stance number for agent {agent.agent_id} "
                f"after {max_retries + 1} attempts on sample {i+1}. Last error: {last_error}"
            )

    mean_value = sum(samples) / len(samples)
    return mean_value, first_raw


def get_initial_stance(
    client: LLMClient,
    agent: Agent,
    topic_context: str,
    stance_question: str,
    frozen_value: float = None,
) -> tuple:
    if frozen_value is not None:
        return frozen_value, f"(frozen initial stance: {frozen_value})"
    return elicit_stance(client, agent, topic_context, stance_question, [])