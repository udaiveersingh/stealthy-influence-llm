# Stealthy Influence in LLM-Mediated Communities

**Status: Steps 1–5 complete (organic baseline, validated across 8 topic categories). Step 6 (adversary) next.**

## Setup

1. Get an NVIDIA API key: https://build.nvidia.com
2. Install dependencies:
   ```
   pip install openai
   ```
   (Groq's SDK is also present for historical/comparison runs, but NVIDIA is now the primary and only actively-used backend — see "Backend history" below.)
3. Set your API key and provider:
   ```powershell
   $env:MODEL_PROVIDER="nvidia"
   $env:NVIDIA_API_KEY="your_key_here"
   ```

## Run order

```powershell
# Validate the backend works at all (cheap, 1 trial)
python -m src.test_nvidia_backend

# Run the full organic baseline across all 8 topics (5 trials each)
python -m src.batch_organic

# Analyze results: moderation, convergence, polarization, conformity — split by topic and model
python -m src.analyze_metrics

# Persona-level breakdown: does persona type predict reactivity on sensitive vs. practical topics?
python -m src.analyze_persona_topic
```

If a batch run gets interrupted (e.g. rate limiting), just re-run `batch_organic.py` — it skips already-completed trials automatically by scanning file contents, not filenames.

## Project structure

```
config/
  personas.json          -- 8 agent definitions: 6 ordinary personas, 1 high-status
                             (with observable status label), 1 adversary (currently
                             behaves as an ordinary agent in organic trials, for
                             population parity with future adversarial conditions)
  topics.json             -- 8 topics across 8 categories: social/lifestyle,
                             technology/policy, education, human values,
                             relationships/society, religion/belief,
                             politics/governance, ethics/morality

src/
  schemas.py              -- Agent, Message, TrialRecord data structures.
                             Every trial stamps model_provider/model_name/temperature.
  llm_client.py            -- Provider-agnostic client (Groq or NVIDIA via
                             MODEL_PROVIDER env var). Handles NVIDIA rate-limit
                             backoff and the enable_thinking:False fix (see below).
  stance.py                -- Stance elicitation. Requires an explicit STANCE:
                             marker in the response; no unsafe "grab any number"
                             fallback. Retries on parse failure.
  conversation.py           -- Conversation round generation. Status is observable
                             to other agents. Robust POST: marker extraction with
                             placeholder/truncation detection and retry.
  organic_pipeline.py       -- Full trial pipeline: pre-stance -> N rounds -> post-stance
                             -> logging. 8-agent population parity fixed. Supports
                             frozen initial stances for future paired trial designs.
  batch_organic.py          -- Runs N trials x all 8 topics, resumable, per-trial
                             error resilience (one failed trial doesn't kill the batch).
  analyze_metrics.py         -- Computes 5 metrics per trial: signed shift, moderation,
                             convergence, group polarization, conformity. Splits
                             results by (topic, model_provider) automatically.
  analyze_persona_topic.py   -- Per-persona breakdown: sensitive vs. practical topics,
                             absolute vs. signed shift (don't confuse the two --
                             see in-script notes).

  test_nvidia_backend.py     -- One-trial sanity check for the NVIDIA backend.
  test_reasoning_control.py  -- Isolated test proving enable_thinking:False works
                             (kept for reference/reproducibility).
  debug_stance_direction.py  -- Debug tool: prints raw stance completions to check
                             for truncation/sign issues.
  cleanup_debug_trials.py    -- Removes ad-hoc debug trials (seed=9001) from logs/
                             so they don't contaminate batch statistics.
```

## Backend history (for the record / presentation)

The project started on **Groq (gpt-oss-20b)**, which validated the whole pipeline (Steps 1-5) but hit a hard 200,000 token/day free-tier wall that couldn't support the trial volume needed for 8 topics x multiple conditions. Switched to **NVIDIA (Nemotron 3.5 Lightning 30B)**, which required its own round of fixes:

- Nemotron's extended reasoning caused massive truncation and reasoning-leak bugs at first (fixed via stricter output-format markers, retry-on-placeholder-detection, and eventually `enable_thinking:False` via `extra_body`, confirmed ~20-25x faster with clean output).
- NVIDIA's per-minute rate limit was hit once reasoning got fast (429 errors) — fixed with a proactive throttle plus automatic backoff/retry.

Groq is retired from active use; NVIDIA is now the sole backend for all data collection going forward.

## Known limitations / caveats (be upfront about these)

- Two known outlier trials (`remote_work` seed=2004, `centralized_governance` seed=2004) show extreme whole-group swings. Both were manually inspected and are legitimate (coherent conversations, not degenerate output) but disproportionately affect aggregate statistics at n=5 — `analyze_persona_topic.py` can exclude them for a cleaner read.
- "Sensitive" vs. "practical" topic categorization in `analyze_persona_topic.py` is a manual researcher choice, not derived from data.
- n=5 trials/topic is enough to see patterns but not enough for strong statistical claims — treat single-persona or single-topic findings as suggestive.
- Occasional stance/message content mismatches observed (~rare) where an agent's recorded numeric stance doesn't perfectly match its qualitative message content — worth a footnote in any writeup, not yet systematically fixed.

## Next: Step 6 — the adversary

Build the first influence mechanism (disinformation, manufactured consensus, or personalized persuasion — see project proposal for the full taxonomy) now that the organic baseline is validated across all 8 topic categories.