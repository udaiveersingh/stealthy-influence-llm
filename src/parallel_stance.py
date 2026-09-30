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
    pinned_prefix: list = None,
) -> dict:
    def one(agent):
        value, _ = elicit_stance(
            client, agent, topic_context, stance_question, conversation_history,
            pinned_prefix=pinned_prefix,
        )
        return agent.agent_id, value

    with ThreadPoolExecutor(max_workers=min(max_workers, len(agents))) as ex:
        results = list(ex.map(one, agents))
    return dict(results)