"""
Standalone mock client for offline pilot testing ONLY.
Lives outside src/ and under a different name than your real LLMClient so it
never collides with or gets confused for src/llm_client.py (which validates
provider as 'groq' or 'nvidia' and expects real credentials).

Use this to sanity-check the round loop / attacker wiring for free. Swap it
for your real src.llm_client.LLMClient (provider="nvidia") when you're ready
to run for real -- see the note in pilot_demo.py.
"""
import re


class MockLLMClient:
    def __init__(self, provider="mock", model="mock-model"):
        self.provider = provider
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.9,
                 max_tokens: int = 1800) -> str:
        agent_hint = "none"
        m = re.search(r"You are responding to (\w+)", user_prompt)
        if m:
            agent_hint = m.group(1)
        return f"POST: This is a mock reply for testing purposes (system_len={len(system_prompt)}, target={agent_hint})."

    def info(self):
        return {"provider": self.provider, "model": self.model}