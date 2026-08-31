"""
Thin wrapper around the Groq API.

Requires the environment variable GROQ_API_KEY to be set.
Get a free key at https://console.groq.com
"""
import os
from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set. "
                "Get a free key at https://console.groq.com and run:\n"
                "  export GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.8, max_tokens: int = 200) -> str:
        """Single-turn completion. Returns the raw text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
