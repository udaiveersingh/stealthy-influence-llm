"""
Core data structures for the simulation.
Keep this schema stable -- every downstream metric (stance shift, persistence,
detection features) is just a function over a TrialRecord.
"""
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Agent:
    agent_id: str
    persona_description: str
    is_high_status: bool = False
    is_adversary: bool = False


@dataclass
class Message:
    agent_id: str
    round_number: int
    content: str
    timestamp: float = field(default_factory=time.time)
    target_agent_id: Optional[str] = None  # who this message is "replying to", if applicable


@dataclass
class TrialRecord:
    trial_id: str
    condition: str          # "organic" | "disinformation" | "manufactured_consensus" | "personalized"
    topic_id: str
    seed: int
    messages: list = field(default_factory=list)          # list[Message]
    pre_stances: dict = field(default_factory=dict)        # agent_id -> float
    post_stances: dict = field(default_factory=dict)       # agent_id -> float (after intervention)
    post_removal_stances: dict = field(default_factory=dict)  # agent_id -> float (after attacker removed), adversarial only

    def to_dict(self):
        return {
            "trial_id": self.trial_id,
            "condition": self.condition,
            "topic_id": self.topic_id,
            "seed": self.seed,
            "messages": [vars(m) for m in self.messages],
            "pre_stances": self.pre_stances,
            "post_stances": self.post_stances,
            "post_removal_stances": self.post_removal_stances,
        }
