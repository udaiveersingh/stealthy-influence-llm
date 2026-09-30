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

Rate limiting (this version): after enable_thinking:False made calls ~20x
faster we started hitting NVIDIA's per-minute limit (429). Two things made
that unrecoverable once the pipeline went parallel, and both are fixed here:

  1. NO GLOBAL CAP. Each cell runs up to 8 stance threads, and --parallel 4
     runs 4 cells, so 4 organic controls starting together put 32 requests in
     flight at once. A module-level semaphore now caps requests in flight
     across ALL threads and ALL client instances (default 8; override with
     LLM_MAX_CONCURRENCY). It wraps only the HTTP call, never the backoff
     sleep, so a thread waiting to retry doesn't hold a slot. A separate global
     requests-per-minute limiter (LLM_MAX_RPM, default 40) covers the case
     where the limit is a rate rather than a concurrency count.

  2. NO JITTER. Every thread retried on the same 3/6/12/20/30s schedule, so
     blocked threads woke together and re-hit the limit together. Waits now
     get +/-50% random jitter to spread the retries out.
"""
import os
import random
import threading
import time

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BACKOFF_SECONDS = [3, 6, 12, 20, 30, 45]  # increasing wait between retries
BACKOFF_JITTER = 0.5            # each wait is scaled by uniform(1-J, 1+J); 0 disables
PROACTIVE_THROTTLE_SECONDS = 0.6

_MAX_IN_FLIGHT = int(os.environ.get("LLM_MAX_CONCURRENCY", "8"))
_request_slots = threading.BoundedSemaphore(_MAX_IN_FLIGHT)

# Global request-RATE limit (requests started per minute, across all threads and
# client instances). A concurrency cap alone doesn't bound a per-minute limit:
# with a fast API, 8 slots can still start hundreds of requests a minute. The
# default is a guess at a conservative free-tier figure -- I could not verify
# NVIDIA's actual limit for your account. Lower it if you still see 429s; raise
# it (or set 0 to disable) if you have a higher quota. Your earlier runs averaged
# ~3.5 requests/min, so this default doesn't slow anything you've run so far.
_MAX_RPM = float(os.environ.get("LLM_MAX_RPM", "40"))
_rate_lock = threading.Lock()
_next_start = 0.0


def set_max_rpm(rpm: float) -> None:
    global _MAX_RPM, _next_start
    _MAX_RPM = rpm
    _next_start = 0.0


def _wait_for_rate_slot() -> None:
    """Reserve the next request start time and sleep until it. Called before
    EVERY attempt including retries -- retries count against a rate limit too,
    so an unthrottled retry storm keeps the window full and prevents recovery."""
    global _next_start
    if _MAX_RPM <= 0:
        return
    interval = 60.0 / _MAX_RPM
    with _rate_lock:
        now = time.monotonic()
        start = max(now, _next_start)
        _next_start = start + interval
    delay = start - now
    if delay > 0:
        time.sleep(delay)


def set_max_concurrency(n: int) -> None:
    """Replace the process-wide in-flight cap. Call before any requests start."""
    global _request_slots
    _request_slots = threading.BoundedSemaphore(n)


MAX_RETRY_AFTER_SECONDS = 120


def _retry_after_seconds(exc):
    """Seconds the server asked us to wait (Retry-After header), or None."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return min(float(headers.get("retry-after")), MAX_RETRY_AFTER_SECONDS)
    except (TypeError, ValueError):
        return None


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
            self.client = Groq(api_key=api_key, max_retries=0)
            self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

        elif self.provider == "nvidia":
            from openai import OpenAI
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "NVIDIA_API_KEY environment variable not set. "
                    "Get a key at https://build.nvidia.com"
                )
            # max_retries=0: the SDK's default is 2 automatic retries on 429, i.e.
            # ONE create() call can send THREE HTTP requests. That multiplies the
            # request rate exactly when the server is already rejecting us, and
            # is invisible to the rate limiter below. All retrying happens in
            # complete() instead, where it is counted, throttled, and printed.
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                max_retries=0,
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

    def _is_transient_error(self, exc: Exception) -> bool:
        """Retryable errors: rate limits PLUS transient server/network failures.
        Previously only 429 was retried, so a single 502/503/504 or a dropped
        connection anywhere in a trial killed it and discarded every call the
        trial had already completed -- at ~2 requests/min a trial takes ~30 min,
        so one transient blip threw away half an hour of work. These errors say
        nothing about the request itself; retrying is the correct response."""
        if self._is_rate_limit_error(exc):
            return True
        msg = str(exc).lower()
        if any(f"error code: {c}" in msg for c in ("500", "502", "503", "504")):
            return True
        name = type(exc).__name__.lower()
        return any(k in name for k in ("timeout", "connection", "internalserver", "serviceunavailable"))

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
            time.sleep(PROACTIVE_THROTTLE_SECONDS)

        last_exc = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                _wait_for_rate_slot()
                with _request_slots:
                    response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self._is_transient_error(e) and attempt < RATE_LIMIT_MAX_RETRIES:
                    base = RATE_LIMIT_BACKOFF_SECONDS[min(attempt, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)]
                    wait = base * random.uniform(1 - BACKOFF_JITTER, 1 + BACKOFF_JITTER)
                    server_wait = _retry_after_seconds(e)
                    note = ""
                    if server_wait is not None and server_wait > wait:
                        wait, note = server_wait, " (server Retry-After)"
                    kind = "hit 429" if self._is_rate_limit_error(e) else f"server error ({type(e).__name__})"
                    print(f"    [rate limit] {kind}, waiting {wait:.0f}s before retry "
                          f"({attempt + 1}/{RATE_LIMIT_MAX_RETRIES}){note}...")
                    time.sleep(wait)
                    last_exc = e
                    continue
                raise  # not a transient error, or retries exhausted -- propagate as before

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