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

# Open-advocate condition (run_confirmatory.py --disclose): agent_id -> public role label shown next to that
# agent's id in every prompt other agents see (posting history and stance questions). Empty in all other runs.
DISCLOSED = {}


def display_id(agent_id: str) -> str:
    base = agent_id.split("_")[0] if NEUTRAL else agent_id
    if agent_id in DISCLOSED:
        return f"{base} ({DISCLOSED[agent_id]})"
    return base