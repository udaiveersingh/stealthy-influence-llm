# Update Package: NVIDIA backend + three methodology fixes + conformity metric

## What changed and why

### 1. Provider-agnostic LLM client (`src/llm_client.py` — full rewrite)
Now supports both Groq and NVIDIA via one interface. Switch with env vars, no code changes:

```powershell
# Groq (original)
$env:MODEL_PROVIDER="groq"
$env:GROQ_API_KEY="your_key"

# NVIDIA (new)
$env:MODEL_PROVIDER="nvidia"
$env:NVIDIA_API_KEY="your_key"
```

Deliberately does NOT set `reasoning_budget` / `enable_thinking` for NVIDIA models, per instruction — avoids repeating the gpt-oss empty-content bug.

### 2. Three methodology fixes (all required before Step 6 / the adversary)

- **8-vs-8 parity**: `load_agents()` in `organic_pipeline.py` now includes all 8 agents by default (the adversary just behaves normally in organic trials). Previously excluded, giving 7 vs. 8 — a real confound.
- **Observable high-status signal**: `conversation.py` now shows other agents a visible tag like `[A7_STATUS - Respected senior community member]`, not just the plain agent ID. Verified working (see test output).
- **Frozen initial-stance support**: `stance.py`'s new `get_initial_stance()` accepts an optional fixed value per agent, skipping fresh elicitation. Ready for paired trial designs later.

### 3. Conformity metric (`src/analyze_metrics.py`)
Added per your professor's formula: does an agent move toward the group's *initial* position, regardless of whether that's toward or away from zero? This is different from moderation and directly tests the conformity-literature question (Choi et al. 2025, Zhu et al. 2025).

### 4. Model tracking (`src/schemas.py`)
Every trial now records `model_provider`, `model_name`, and `temperature`. `analyze_metrics.py` and `batch_organic.py` both split results by provider automatically — gpt-oss and Nemotron data can never get silently mixed.

### 5. Expanded topics (`config/topics.json`)
8 categories now, per your professor's request: social/lifestyle, technology/policy, education, human values, relationships/society, religion/belief, politics/governance, ethics/morality. Original 3 topics unchanged (same IDs) for continuity with existing data. **The 5 new topics are configured but not yet run — that's Phase 3, after backend validation.**

### 6. Bonus fix: synthetic timestamps
`conversation.py` now assigns synthetic `sim_time` (round × 100 + position) instead of relying solely on wall-clock time, which would otherwise measure API latency, not real social timing. This matters later for detection features (Step 9), doesn't affect anything now.

## Exact run order

**Step A — validate Nemotron works at all (small, cheap):**
```powershell
$env:MODEL_PROVIDER="nvidia"
$env:NVIDIA_API_KEY="your_key"
python -m src.test_nvidia_backend
```
Runs ONE trial: 8 agents, 3 rounds, 1 topic. Read the printed messages — do personas sound distinct and sensible? Check the saved JSON for the status label appearing correctly in context.

**Step B — once Nemotron looks sane, run the fresh corrected baseline on BOTH backends:**
```powershell
# gpt-oss (Groq) — fresh seeds (2000+), 8-agent parity, no duplicates
$env:MODEL_PROVIDER="groq"
python -m src.batch_organic

# Nemotron (NVIDIA) — same seeds, same topics, for direct comparison
$env:MODEL_PROVIDER="nvidia"
python -m src.batch_organic
```
This gives you clean 5-trial-per-topic data under BOTH models, all with 8-agent parity and observable status — nothing here overlaps with your old (7-agent, non-parity) dataset, since seeds start at 2000.

**Step C — compare:**
```powershell
python -m src.analyze_metrics
```
Now splits automatically by (topic, model_provider). Look for: does Nemotron produce *qualitatively similar* convergence/moderation/polarization/conformity patterns to gpt-oss? That's the actual question — not "which model is better."

## What's intentionally NOT done yet

- The 5 new topic categories are configured but not run (Phase 3, after backend comparison).
- Persona × topic analysis (does empathetic vs. pragmatic behave differently on religion) — this needs the topic expansion done first.
- The adversary (Step 6) — still next after this validation pass.
- Your old 7-agent dataset (23 trials) is untouched in `logs/` — keep it as pilot/validation data, but don't mix it into the fresh 8-agent comparison.
