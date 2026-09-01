"""
Thin wrapper around the Groq API.

Requires the environment variable GROQ_API_KEY to be set.
Get a free key at https://console.groq.com
"""
import os
from groq import Groq

# Default can be overridden by setting GROQ_MODEL env var, e.g.:
#   $env:GROQ_MODEL="openai/gpt-oss-20b"   (PowerShell)
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


class LLMClient:
    def __init__(self, model: str = None):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set. "
                "Get a free key at https://console.groq.com and run:\n"
                "  $env:GROQ_API_KEY='your_key_here'"
            )
        self.client = Groq(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.8, max_tokens: int = 400) -> str:
        """Single-turn completion. Returns the raw text response."""
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # gpt-oss models are reasoning models: they spend tokens on hidden
        # chain-of-thought before producing visible content. Ask for low
        # reasoning effort so more of max_tokens goes to the actual answer.
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = "low"

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None or content.strip() == "":
            raise RuntimeError(
                f"Model '{self.model}' returned empty content. Full response object: {response}"
            )
        return content.strip()