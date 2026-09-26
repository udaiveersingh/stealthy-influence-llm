"""
Parallel stance elicitation.

Why this exists: a trial is 56 API calls -- 16 pre-stance, 24 conversation,
16 post-stance. The 24 conversation calls MUST stay sequential (each agent
has to see the previous agent's message; that dependency is the whole
design). But the 32 stance calls are completely independent of each other:
agent A's stance doesn't depend on agent B's.

Measured on your setup: 4 concurrent calls overlapped cleanly (~4x), so
server latency is wait time, not compute contention.

Effect per trial, at a ~15s median call:
    before:  56 sequential calls          ~14 min
    after:   24 conv + ~2 stance batches  ~ 7 min

This does NOT change any stance logic. It calls your existing
elicit_stance() / get_initial_stance() once per agent, just concurrently.
All retry, parsing, sampling and frozen-value behaviour is unchanged, so
results are identical to the sequential path.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.stance import elicit_stance, get_initial_stance

DEFAULT_WORKERS = 8


def get_initial_stances_parallel(
    client,
    agents: list,
    topic_context: str,
    stance_question: str,
    frozen_initial_stances: dict = None,
    max_workers: int = DEFAULT_WORKERS,
) -> dict:
    """Pre-stance for every agent, concurrently. Frozen values short-circuit
    inside get_initial_stance without an API call, exactly as before."""
    frozen = frozen_initial_stances or {}

    def one(agent):
        value, _ = get_initial_stance(
            client, agent, topic_context, stance_question, frozen.get(agent.agent_id)
        )
        return agent.agent_id, value

    with ThreadPoolExecutor(max_workers=min(max_workers, len(agents))) as ex:
        results = list(ex.map(one, agents))
    return dict(results)


def elicit_stances_parallel(
    client,
    agents: list,
    topic_context: str,
    stance_question: str,
    conversation_history: list,
    max_workers: int = DEFAULT_WORKERS,
) -> dict:
    """Post-stance for every agent, concurrently. Every agent sees the same
    completed conversation, so there is no ordering dependency here."""

    def one(agent):
        value, _ = elicit_stance(
            client, agent, topic_context, stance_question, conversation_history
        )
        return agent.agent_id, value

    with ThreadPoolExecutor(max_workers=min(max_workers, len(agents))) as ex:
        results = list(ex.map(one, agents))
    return dict(results)
