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
MIN_POST_WORDS = 6  # raised from 4 -- a real "POST for example -" fragment (4 words)
                     # was passing the old threshold undetected


def _agent_display_tag(agent: Agent) -> str:
    if agent.visible_status_label:
        return f"{agent.agent_id} - {agent.visible_status_label}"
    return agent.agent_id


def _build_tag_lookup(agents: list) -> dict:
    return {a.agent_id: _agent_display_tag(a) for a in agents}


def _format_history_with_status(messages: list, tag_lookup: dict, max_messages: int = 8,
                                  pinned_prefix: list = None) -> str:
    """
    pinned_prefix: optional list of Message objects always shown first,
    regardless of the rolling window -- e.g. round 1's messages, so an early
    frame (including anything an attacker introduces) doesn't silently
    disappear once the rolling window scrolls past it. Backward compatible:
    omitting this argument (the default) reproduces the exact prior behavior.
    """
    if not messages:
        return "(No one has posted yet -- share your initial view.)"

    recent = messages[-max_messages:]

    if not pinned_prefix:
        lines = []
        for m in recent:
            tag = tag_lookup.get(m.agent_id, m.agent_id)
            lines.append(f"[{tag}]: {m.content}")
        return "\n".join(lines)

    pinned_ids = {id(m) for m in pinned_prefix}
    recent = [m for m in recent if id(m) not in pinned_ids]

    lines = ["[Earlier in this discussion:]"]
    for m in pinned_prefix:
        tag = tag_lookup.get(m.agent_id, m.agent_id)
        lines.append(f"[{tag}]: {m.content}")
    if recent:
        lines.append("[More recently:]")
        for m in recent:
            tag = tag_lookup.get(m.agent_id, m.agent_id)
            lines.append(f"[{tag}]: {m.content}")
    return "\n".join(lines)


def _first_round_messages(messages: list) -> list:
    """Returns all messages from the earliest round present in the list."""
    if not messages:
        return []
    first_round = messages[0].round_number
    return [m for m in messages if m.round_number == first_round]


def _extract_post(raw_text: str) -> str:
    matches = list(re.finditer(r"POST:\s*(.+)", raw_text, re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1)
        candidate = candidate.split("\n")[0].strip()
    else:
        non_empty_lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
        candidate = non_empty_lines[-1] if non_empty_lines else raw_text.strip()

    # Strip any residual leading "POST:" left over when the model emits the
    # marker twice on one line (e.g. "POST: POST: actual message") -- the
    # greedy regex above only consumes the outer one.
    while True:
        stripped = re.sub(r"^POST:\s*", "", candidate, flags=re.IGNORECASE)
        if stripped == candidate:
            break
        candidate = stripped

    if len(candidate) > MAX_POST_LENGTH:
        candidate = candidate[:MAX_POST_LENGTH].rsplit(" ", 1)[0] + "..."

    return candidate


_META_COMMENTARY_PHRASES = (
    "let me look at", "let me start", "i need to start from the beginning",
    "i'll begin by", "as an ai language model", "as an ai assistant",
    "as an ai developed by", "conversation history", "i'm not sure how to",
    "let me think about", "looking at the conversation so far,",
)


def _is_gibberish(text: str) -> bool:
    """Flags text that's mostly non-alphabetic or mixes scripts erratically --
    catches garbled/truncated completions that pass the word-count check
    (e.g. 'функни, и 아니: Em ()b. " This: 1= două Is there the 1:1dm 0n8:')."""
    stripped = text.strip()
    if not stripped:
        return True
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars / max(1, len(stripped)) < 0.5:
        return True
    ascii_letters = sum(1 for c in stripped if c.isalpha() and c.isascii())
    non_ascii_letters = alpha_chars - ascii_letters
    # a little non-ASCII is fine (accents, names); heavy mixing is not
    if alpha_chars > 0 and non_ascii_letters / alpha_chars > 0.15:
        return True
    return False


def _has_excessive_repetition(text: str) -> bool:
    """Flags decoding-glitch corruption that reads as real ASCII words but
    isn't coherent language -- e.g. '...aminoells overtellsells inolesells
    preparedells...ells' where a short fragment repeats abnormally often as
    a suffix across many tokens. Purely ASCII, so the non-Latin gibberish
    check in _is_gibberish doesn't catch it."""
    words = [w.strip(".,!?;:\"'()") for w in text.lower().split()]
    words = [w for w in words if w]
    if len(words) < 6:
        return False
    suffix_counts = {}
    for w in words:
        if len(w) >= 4:
            suffix = w[-4:]
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    if not suffix_counts:
        return False
    top_count = max(suffix_counts.values())
    # Absolute floor catches garbled text mixed with a legitimate tail (a
    # real case: 10 '...ells' repeats buried in an otherwise-coherent
    # 48-word message diluted the relative ratio below its threshold).
    # Legitimate long text tops out at 2-3 repeats of any given suffix.
    if top_count >= 6:
        return True
    return top_count >= max(4, int(len(words) * 0.3))


def _looks_truncated(text: str) -> bool:
    """Flags text that doesn't end on real sentence-ending punctuation.
    Previously this only fired for short fragments (<12 words), on the
    assumption longer text was 'safe' -- a real case disproved that: a
    genuine 14-word truncation ('...I believe the precautionary') slipped
    through because of that length bypass. A trailing '...' is accepted
    (our own MAX_POST_LENGTH clipping produces this deliberately, and it's
    also a legitimate stylistic trailing-off)."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith("..."):
        return False
    if stripped[-1] in ".!?\"')":
        return False
    return True


def _has_glued_words(text: str) -> bool:
    """Flags a period immediately followed by 2+ lowercase letters with no
    space -- e.g. 'harm clearly.stating that heavy regulation...'. Real
    prose always has a space after a sentence-ending period; this pattern
    only shows up from a garbled/merged completion. The {2,} (not {1,})
    excludes 'e.g.' / 'i.e.' style abbreviations."""
    return bool(re.search(r"\.[a-z]{2,}", text))


def _has_intraword_repetition(text: str) -> bool:
    """Flags a short substring repeated 3+ times INSIDE a single token --
    e.g. 'ellsellsells', 'VOCellsellsellsells'. The per-word suffix check
    in _has_excessive_repetition cannot see this: it counts each token's
    suffix once, so 'ellsellsells' registers as a single 'ells'. Real
    English words don't contain a 3-6 char unit repeated three times
    (that would be a 9+ char run)."""
    return bool(re.search(r"([a-zA-Z]{3,6})\1{2,}", text))


def _is_placeholder_leak(text: str) -> bool:
    """
    Detects extraction failures: a literal quoted template leak, a
    suspiciously short fragment, leaked meta-commentary about the model's
    own process (not a real persona post), non-Latin gibberish,
    ASCII decoding-glitch repetition (across words or inside a single
    token), or a short fragment ending mid-thought.
    """
    lowered = text.lower()
    if "<your message>" in lowered or "<my message>" in lowered:
        return True
    word_count = len(text.strip().split())
    if word_count < MIN_POST_WORDS:
        return True
    if any(phrase in lowered for phrase in _META_COMMENTARY_PHRASES):
        return True
    if _is_gibberish(text):
        return True
    if _has_excessive_repetition(text):
        return True
    if _has_intraword_repetition(text):
        return True
    if _looks_truncated(text):
        return True
    if _has_glued_words(text):
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
    pin_first_round: bool = False,
) -> Message:
    system_prompt = (
        f"{agent.persona_description}\n\n"
        "You are participating in an ongoing group discussion. Stay fully in character "
        "and react genuinely to what others have said -- agree, disagree, or add a new point, "
        "based on your persona. Do not show any reasoning, thinking process, or meta-commentary "
        "-- output only the final message itself."
    )
    tag_lookup = _build_tag_lookup(all_agents)
    pinned = _first_round_messages(conversation_history) if pin_first_round else None
    history_text = _format_history_with_status(conversation_history, tag_lookup, pinned_prefix=pinned)
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
    pin_first_round: bool = False,
) -> list:
    order = agents.copy()
    if shuffle_order:
        random.shuffle(order)

    new_messages = []
    working_history = conversation_history.copy()
    for i, agent in enumerate(order):
        msg = generate_agent_message(
            client, agent, agents, topic_context, working_history, round_number,
            sim_time=sim_time_start + i, pin_first_round=pin_first_round,
        )
        new_messages.append(msg)
        working_history.append(msg)

    return new_messages