"""
Provider-agnostic LLM client.

Switch backends via environment variables -- no code changes needed elsewhere.

  Groq:
    $env:MODEL_PROVIDER="groq"
    $env:GROQ_API_KEY="..."

  NVIDIA (OpenAI-compatible endpoint) -- now the primary/only backend in use:
    $env:MODEL_PROVIDER="nvidia"
    $env:NVIDIA_API_KEY="..."
    $env:NVIDIA_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b"   (default if unset)

FIX (this version): after enable_thinking:False made calls ~20x faster, we
started hitting NVIDIA's per-minute rate limit (429 Too Many Requests) --
a limit that was always there but unreachable while calls took 60-80+
seconds each. This is a DIFFERENT kind of limit than Groq's daily token cap
(no "tokens per day" message, just a generic 429). Fixed with automatic
retry + backoff specifically for rate-limit errors, so a 429 pauses and
retries instead of failing the whole trial.
"""
import os
import time

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BACKOFF_SECONDS = [3, 6, 12, 20, 30]  # increasing wait between retries


class LLMClient:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = (provider or os.environ.get("MODEL_PROVIDER", "groq")).lower()

        if self.provider == "groq":
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY environment variable not set. "
                    "Get a free key at https://console.groq.com"
                )
            self.client = Groq(api_key=api_key)
            self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

        elif self.provider == "nvidia":
            from openai import OpenAI
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "NVIDIA_API_KEY environment variable not set. "
                    "Get a key at https://build.nvidia.com"
                )
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
            self.model = model or os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)

        else:
            raise ValueError(
                f"Unknown MODEL_PROVIDER '{self.provider}'. Use 'groq' or 'nvidia'."
            )

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Detects a 429 regardless of exact exception class used by either SDK."""
        msg = str(exc)
        return "429" in msg or "Too Many Requests" in msg or "rate_limit" in msg.lower()

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.8, max_tokens: int = 400) -> str:
        """Single-turn completion. Returns the raw text response. Same interface regardless of provider."""
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        if self.provider == "groq" and "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = "low"

        if self.provider == "nvidia":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            # Small proactive throttle: enable_thinking:False made calls ~20x
            # faster, which is what exposed the RPM limit in the first place.
            # A brief pause between calls means fewer 429s to recover from,
            # which is faster overall than hitting the limit repeatedly.
            time.sleep(0.6)

        last_exc = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < RATE_LIMIT_MAX_RETRIES:
                    wait = RATE_LIMIT_BACKOFF_SECONDS[min(attempt, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)]
                    print(f"    [rate limit] hit 429, waiting {wait}s before retry "
                          f"({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
                    time.sleep(wait)
                    last_exc = e
                    continue
                raise  # not a rate-limit error, or retries exhausted -- propagate as before

            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                raise RuntimeError(
                    f"Model '{self.model}' (provider={self.provider}) returned empty content. "
                    f"Full response object: {response}"
                )
            return content.strip()

        # Only reached if all retries were rate-limit errors
        raise RuntimeError(f"Exhausted {RATE_LIMIT_MAX_RETRIES} retries on rate limiting. Last error: {last_exc}")

    def info(self) -> dict:
        return {"provider": self.provider, "model": self.model}