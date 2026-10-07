"""
Display-only agent labels.

Every agent used to see "[A8_ADVERSARY]" and "[A7_STATUS]" in every prompt, in every condition
(including organic) -- so the attacker was literally labelled as one. With NEUTRAL = True the
suffix is dropped for display ("A8", "A7"); internal IDs, logs and analysis keys are unchanged.
A7 still shows its visible status description, since observable status is a design feature.

Default is False so the legacy pilot behaves exactly as before. The confirmatory study sets it
True once at start-up, for ALL arms, so organic controls and attack arms see identical labels.
"""
NEUTRAL = False


def display_id(agent_id: str) -> str:
    return agent_id.split("_")[0] if NEUTRAL else agent_id