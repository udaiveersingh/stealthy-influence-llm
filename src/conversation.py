"""
A single round of ordinary (non-adversarial) conversation.
Each agent, in turn, sees recent history and posts one message reacting
honestly per its persona.

Status is observable via visible_status_label (shown inline to other agents).

FIX (this version): placeholder/truncation detection now also catches
suspiciously SHORT outputs (e.g. a single word like "Here"), not just literal
"<message>" template leaks. Confirmed real failure mode: a genuine truncated
fragment slipped through the previous detector because it didn't match the
placeholder pattern, even though it was clearly not a real message.
"""
import random
import re
from src.llm_client import LLMClient
from src.schemas import Agent, Message

MAX_POST_LENGTH = 400
MIN_POST_WORDS = 4  # a real 1-2 sentence post should have at least a few words


def _agent_display_tag(agent: Agent) -> str:
    if agent.visible_status_label:
        return f"{agent.agent_id} - {agent.visible_status_label}"
    return agent.agent_id


def _build_tag_lookup(agents: list) -> dict:
    return {a.agent_id: _agent_display_tag(a) for a in agents}


def _format_history_with_status(messages: list, tag_lookup: dict, max_messages: int = 8) -> str:
    if not messages:
        return "(No one has posted yet -- share your initial view.)"
    recent = messages[-max_messages:]
    lines = []
    for m in recent:
        tag = tag_lookup.get(m.agent_id, m.agent_id)
        lines.append(f"[{tag}]: {m.content}")
    return "\n".join(lines)


def _extract_post(raw_text: str) -> str:
    matches = list(re.finditer(r"POST:\s*(.+)", raw_text, re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1)
        candidate = candidate.split("\n")[0].strip()
    else:
        non_empty_lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
        candidate = non_empty_lines[-1] if non_empty_lines else raw_text.strip()

    if len(candidate) > MAX_POST_LENGTH:
        candidate = candidate[:MAX_POST_LENGTH].rsplit(" ", 1)[0] + "..."

    return candidate


def _is_placeholder_leak(text: str) -> bool:
    """
    Detects extraction failures: either a literal quoted template leak
    (e.g. "<my message>"), OR a suspiciously short fragment that's clearly
    not a real 1-2 sentence post -- confirmed real case: a single word like
    "Here" left over from a truncated completion.
    """
    lowered = text.lower()
    if "<your message>" in lowered or "<my message>" in lowered:
        return True
    word_count = len(text.strip().split())
    if word_count < MIN_POST_WORDS:
        return True
    return False


def generate_agent_message(
    client: LLMClient,
    agent: Agent,
    all_agents: list,
    topic_context: str,
    conversation_history: list,
    round_number: int,
    sim_time: int,
) -> Message:
    system_prompt = (
        f"{agent.persona_description}\n\n"
        "You are participating in an ongoing group discussion. Stay fully in character "
        "and react genuinely to what others have said -- agree, disagree, or add a new point, "
        "based on your persona. Do not show any reasoning, thinking process, or meta-commentary "
        "-- output only the final message itself."
    )
    tag_lookup = _build_tag_lookup(all_agents)
    history_text = _format_history_with_status(conversation_history, tag_lookup)
    user_prompt = (
        f"{topic_context}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        "Write your next message in this discussion: 1-2 short sentences, natural "
        "social-media-style post.\n\n"
        "Respond in EXACTLY this format, with nothing before or after it:\n"
        "POST: <your message>"
    )

    max_retries = 2
    content = None
    for attempt in range(max_retries + 1):
        raw = client.complete(system_prompt, user_prompt, temperature=0.9, max_tokens=1800)
        candidate = _extract_post(raw)
        if _is_placeholder_leak(candidate):
            print(f"    [retry {attempt+1}/{max_retries+1}] {agent.agent_id}: "
                  f"got placeholder/truncated leak (raw length={len(raw)} chars, "
                  f"extracted={candidate!r}) -- retrying")
            continue
        content = candidate
        break

    if content is None:
        print(f"    [WARNING] {agent.agent_id}: all {max_retries+1} attempts failed to produce "
              f"a real message -- storing explicit placeholder instead of leaked reasoning")
        content = "(no substantive response generated)"

    return Message(agent_id=agent.agent_id, round_number=round_number, content=content, sim_time=sim_time)


def run_organic_round(
    client: LLMClient,
    agents: list,
    topic_context: str,
    conversation_history: list,
    round_number: int,
    shuffle_order: bool = True,
    sim_time_start: int = 0,
) -> list:
    order = agents.copy()
    if shuffle_order:
        random.shuffle(order)

    new_messages = []
    working_history = conversation_history.copy()
    for i, agent in enumerate(order):
        msg = generate_agent_message(
            client, agent, agents, topic_context, working_history, round_number,
            sim_time=sim_time_start + i,
        )
        new_messages.append(msg)
        working_history.append(msg)

    return new_messages