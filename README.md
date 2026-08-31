# Stealthy Influence — Project Skeleton (Steps 1-3)

## Setup

1. Get a free Groq API key: https://console.groq.com
2. Install dependencies:
   ```
   pip install groq
   ```
3. Set your API key (don't hardcode it anywhere):
   ```
   export GROQ_API_KEY=your_key_here
   ```

## Run the sanity check

```
python3 test_skeleton.py
```

This will:
- Elicit pre-conversation stances from all 7 ordinary agents (adversary excluded for now)
- Run one organic conversation round
- Elicit post-round stances and show the shift
- Print everything so you can eyeball whether it's behaving sanely

## What to check when you run this

- **Does every stance elicitation parse successfully?** If you see retries/failures in the console, the prompt or parser needs tuning.
- **Do the posted messages actually sound like their personas?** A1 should sound cautious/data-driven, A4 should sound emotional/story-driven, etc. If everyone sounds the same, the persona prompts need to be more distinct.
- **Is the model reliably returning numbers in range?** Occasional outliers are fine (they get clamped), but if it's ignoring the instruction entirely, the prompt needs rework.

## Project structure

```
config/
  personas.json   -- 8 agent definitions (edit personas/topics here, not in code)
  topics.json      -- 3 topics with stance-elicitation questions
src/
  schemas.py       -- Agent, Message, TrialRecord data structures
  llm_client.py    -- Groq API wrapper
  stance.py        -- stance elicitation + parsing
  conversation.py  -- single organic conversation round
test_skeleton.py   -- sanity-check script (Steps 1-3 validation)
```

## Next steps (not yet built)

- Step 4: full organic baseline pipeline (multiple rounds, full trial logging via TrialRecord)
- Step 5: run organic trials at volume, check stance-shift distribution has enough spread
- Step 6: adversary agent logic (manufactured consensus mechanism first)
- Step 7: adversarial pipeline with persistence measurement (post-removal stance)
- Steps 8-10: remaining mechanisms, detection features, statistical analysis

Run `test_skeleton.py` first and report back what you see — especially any parsing failures or personas that don't sound distinct — before we build Step 4.
