"""
Stance elicitation: ask an agent (in persona) for its current numeric stance
on a topic, given whatever conversation history it has seen so far.

FIX (this version, root-caused via debug_stance_direction.py): the previous
fallback -- "if no STANCE: marker found, grab the last number anywhere in
the text" -- was unsafe. A real case: the model's completion got truncated
mid-reasoning (350 tokens wasn't enough) right after it wrote out its OWN
scale legend ("0: neutral/undecided"), and the fallback grabbed that "0" as
if it were a real answer. This silently produced a wrong stance value that
looked valid but wasn't -- and directly contradicted the agent's own
conversation history.

Fix: (1) raised max_tokens substantially, same reasoning as the message-
generation fix -- this model's reasoning can run long and needs headroom to
actually reach its answer. (2) REMOVED the unsafe fallback entirely: if no
STANCE: marker is found, that's now treated as a parse failure and retried,
not silently guessed at.
"""
import re
from src.llm_client import LLMClient
from src.schemas import Agent, Message


def _format_history(messages: list, max_messages: int = 10) -> str:
    if not messages:
        return "(No conversation has happened yet.)"
    recent = messages[-max_messages:]
    lines = [f"[{m.agent_id}]: {m.content}" for m in recent]
    return "\n".join(lines)


def _parse_stance_number(raw_text: str) -> float:
    """
    Requires the explicit STANCE: <number> marker. Takes the LAST match if
    there are multiple (a model can quote the marker back to itself mid-
    reasoning before giving its real final answer).

    Does NOT fall back to "any number in the text" -- that was proven unsafe:
    it can grab a number from the model's own scale-legend explanation
    (e.g. "0: neutral") rather than an actual chosen answer. If the marker
    is genuinely absent (usually means truncation before the model reached
    its answer), this raises ValueError so the caller retries with a fresh
    sample instead of silently accepting a guessed value.
    """
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
) -> tuple:
    system_prompt = (
        f"{agent.persona_description}\n\n"
        "You are participating in a group discussion. Stay fully in character. "
        "Do not show any reasoning, thinking process, or meta-commentary -- respond "
        "with only the requested final answer, nothing else."
    )
    history_text = _format_history(conversation_history)
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
            # Raised from 350 -> 1200: same root cause as the message-generation
            # fix -- this model's reasoning can run long, and needs enough
            # headroom to actually reach the STANCE: line, not just explain
            # the scale and get cut off.
            raw = client.complete(system_prompt, user_prompt, temperature=0.3, max_tokens=1200)
            try:
                value = _parse_stance_number(raw)
                samples.append(value)
                if first_raw is None:
                    first_raw = raw
                break
            except ValueError as e:
                last_error = e
                print(f"    [stance retry {attempt+1}/{max_retries+1}] {agent.agent_id}: "
                      f"{e}")
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