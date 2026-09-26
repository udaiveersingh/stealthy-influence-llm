"""
Standalone mock client for offline pilot testing ONLY.
Lives outside src/ and under a different name than your real LLMClient so it
never collides with src/llm_client.py (which validates provider as 'groq' or
'nvidia' and expects real credentials).

Handles BOTH prompt types the pipeline issues:
  - "POST: <message>" conversational turns
  - "STANCE: <number>" stance elicitation

Returns varied text per call so the near-duplicate retry in adversary.py
doesn't fire spuriously during offline testing (a fixed canned string would
look like verbatim self-repetition to that check).
"""
import random
import re


class MockLLMClient:
    def __init__(self, provider="mock", model="mock-model", seed=0):
        self.provider = provider
        self.model = model
        self._rng = random.Random(seed)
        self._n = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.9,
                 max_tokens: int = 1800) -> str:
        self._n += 1

        if "STANCE:" in user_prompt:
            value = round(self._rng.uniform(-1.0, 1.0), 2)
            return f"STANCE: {value}\nREASON: mock reasoning for testing purposes."

        agent_hint = "none"
        m = re.search(r"You are responding to (\w+)", user_prompt)
        if m:
            agent_hint = m.group(1)

        topics = ["oversight", "innovation", "safety", "compliance", "governance",
                  "transparency", "accountability", "deployment"]
        a = self._rng.choice(topics)
        b = self._rng.choice(topics)
        return (f"POST: Mock reply number {self._n} discussing {a} and {b} "
                f"in some detail for testing (target={agent_hint}).")

    def info(self):
        return {"provider": self.provider, "model": self.model}
