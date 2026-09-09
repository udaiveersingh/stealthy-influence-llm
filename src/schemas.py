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
    visible_status_label: Optional[str] = None  # shown to OTHER agents if set -- makes status observable


@dataclass
class Message:
    agent_id: str
    round_number: int
    content: str
    timestamp: float = field(default_factory=time.time)
    sim_time: Optional[int] = None  # synthetic round/slot index -- use this for burstiness, not wall-clock timestamp
    target_agent_id: Optional[str] = None


@dataclass
class TrialRecord:
    trial_id: str
    condition: str          # "organic" | "disinformation" | "manufactured_consensus" | "personalized"
    topic_id: str
    seed: int
    model_provider: str = ""     # "groq" | "nvidia" -- filled from LLMClient.info()
    model_name: str = ""         # exact model string used for this trial
    temperature: Optional[float] = None
    frozen_initial_stances: Optional[dict] = None  # if set, these overrode fresh pre-elicitation
    messages: list = field(default_factory=list)
    pre_stances: dict = field(default_factory=dict)
    post_stances: dict = field(default_factory=dict)
    post_removal_stances: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "trial_id": self.trial_id,
            "condition": self.condition,
            "topic_id": self.topic_id,
            "seed": self.seed,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "frozen_initial_stances": self.frozen_initial_stances,
            "messages": [vars(m) for m in self.messages],
            "pre_stances": self.pre_stances,
            "post_stances": self.post_stances,
            "post_removal_stances": self.post_removal_stances,
        }
