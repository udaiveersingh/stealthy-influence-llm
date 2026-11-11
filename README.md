# Stealthy Influence in LLM-Mediated Communities

Can a small, hidden group of coordinated AI agents shift what a group of AI agents believes, does the shift last after they stop, and can anyone tell it happened?

This repository holds the simulator, the experiments and the analysis behind that question. We put LLM agents in group discussions on policy topics, secretly instruct some of them to push one side, and compare every attacked group with an identical twin that has no attackers.

**Status (November 2026):** all experiments and analyses are complete. The one remaining step is human validation by two to three more independent raters (one blind rater done). A paper is in preparation.

---

## Headline results

Stance is measured on a scale from −1 to +1. "Effect" is how far a group moved toward the attackers' target, minus how far its attacker-free twin moved.

| Question | Result |
|---|---|
| **Does the attack work?** | Yes, in all 24 model × topic settings. Pooled effect **+0.170** [95% CI +0.141, +0.199], p = 2×10⁻¹¹. Prediction interval for a new topic or model [+0.067, +0.274] excludes zero. The three models do not differ (p = .48). |
| **Does it last?** | Partly. **+0.075** remains after three silent rounds, about **44%** [32%, 54%] of the shift. Which topics keep it is **not** shared across models (Spearman −0.50, +0.19, 0.00); backfire happens only on Nemotron (remote work, centralized governance). No tested group property explains persistence. |
| **What happens at scale?** | 32 agents: 8 attackers (+0.149) match 4 of 8; 4 attackers give a small, steady +0.096 that mostly lasts. 512-agent network: 10% attackers give +0.030 at the end and +0.023 after silence (7/8 topics, 3 independent networks), concentrated in agents **directly connected** to an attacker; agents two steps away do not move. 2% attackers: no overall effect. |
| **Can it be caught?** | An LLM auditor detects attacked conversations above chance (AUC up to 0.80 on its own model, 0.87 cross-model) and keeps working when every message is cut to 120 characters (0.78 → 0.79). Auditor strength matters more than matching the attackers' model. But when only **1 in 100** conversations is attacked, the best auditor is right on only **6.7%** of its alarms (about 14 false alarms per real catch). |
| **Do attacker messages really argue their side?** | Blind human labels: **93%** of attacker messages vs **26%** of organic messages from the same agents argue the target side (Fisher p < 10⁻⁸). Human–judge agreement κ = 0.63 (toward vs not). |

Scope: every claim is about **populations of LLM agents**. LLM agents update toward peers more readily than people do, so these effect sizes should not be read as human susceptibility.

---

## Research questions

1. **RQ1 Effect.** Do covert attackers shift group stance toward their target across model families and topics?
2. **RQ2 Persistence.** How much of the shift remains after the attackers go silent, and is the same topic persistent on every model?
3. **RQ3 Scale.** How does the effect change from 8-agent groups to 32-agent groups and a 512-agent network, and how far does it spread?
4. **RQ4 Detectability.** Can text features, an agent-level fingerprint or an LLM auditor tell attacked conversations from organic ones, under length control, across auditor models, and at realistic base rates?

---

## Experimental design

**Protocol (every setting).** Stances are read before round 1 and frozen, so the attacked group and its twin start from the same opinions. Attackers are active in rounds 1–3 and silent in rounds 4–6. Stances are read again after rounds 3 and 6. Effects are measured only on agents who are never attackers.

**Stealth controls on attacker posts.** Messages capped to match ordinary message length (`--match-length`); every attacker draft checked by a direction judge and regenerated if it does not argue the target side (`--enforce-direction`); neutral agent IDs; no shared channel between attackers.

**Models** (all open weights): Nemotron-3.5-Lightning-30B-A3B, gpt-oss-20b, Gemma 4 31B.

**Topics** (8): AI regulation, college value, remote work, free speech vs harm, individual vs collective, traditional family roles, centralized governance, religion in public life. Targets per topic are in `config/topic_plan.json`.

### The three group setups

| | 8-agent group | 32-agent group | 512-agent network |
|---|---|---|---|
| Who talks to whom | One shared chat thread, agents post in turn | Same, one thread | A social network (preferential attachment, median 8 connections, a few hubs) |
| What an agent reads | Round 1 plus the latest messages, so every attacker is visible | Same | A feed: its own last post plus 7 recent posts from its connections; everyone posts at the same time |
| Memory | Round 1 pinned | Round 1 pinned | First feed pinned plus its own last 3 posts ("matched" memory; a "short" variant keeps only the feed) |
| Agents | 8 personas (`config/personas.json`) | 32 personas (`config/personas_scale32.json`) | 32 persona types × 16 copies, each with a random intensity from mild to very strong |
| Attackers | 4 of 8 (also 1 of 8 in some runs) | 4 or 8 of 32 | 10 or 51 of 512 (2%, 10%), random accounts, the 2% set nested in the 10% set |
| Replication | 20 seeds × 8 topics × 3 models | 10 seeds × 8 topics (Nemotron) | 8 topics; 3 independent networks for the 10% arm (Nemotron) |

### Statistics

- Per cell (model × topic): paired differences per seed, t-test and bootstrap CIs (`analyze_confirmatory.py`).
- Pooled: DerSimonian–Laird random effects with Hartung–Knapp CIs and 95% prediction intervals, per model and across all 24 cells; subgroup test for model differences; retention with a cell bootstrap (`pool_confirmatory.py`).
- Society: each topic is one unit (mean over networks), one-sample t-test across the 8 topics; split by network distance to the nearest attacker is exploratory (`analyze_society.py`).
- Mechanism analysis is exploratory (15 uncorrected tests).

---

## Repository layout

```
config/                  personas (8 and 32), topics, per-topic attack targets
src/
  llm_client.py          provider-agnostic client (NVIDIA API or any OpenAI-compatible server via NVIDIA_BASE_URL);
                         rate limiting, retries, reasoning-output recovery
  conversation.py        one round of posts in the shared-thread groups
  stance.py              stance elicitation (explicit STANCE: marker, averaged readings)
  adversary.py           attacker controller: hidden objective, budget, length cap, direction enforcement
  direction_judge.py     judge that labels a post as pushing toward / away from / neither side
  confirmatory.py        one paired trial for the 8- and 32-agent groups
  society.py             the 512-agent network: population, network, feeds, memory rules, parallel rounds, checkpoints
  schemas.py, labels.py, parallel_stance.py, organic_pipeline.py   shared data structures and helpers

run_confirmatory.py      run paired trials (8 or 32 agents) for one topic; resumable via a manifest
run_society.py           run the 512-agent network for one topic (--memory short|matched, --seeds 1 2 3)
analyze_confirmatory.py  per-cell effects, persistence, equivalence tests, figures
pool_confirmatory.py     random-effects pooling across the 24 cells and the 32-agent runs
analyze_society.py       society effects, by network distance, by seed
detectability_confirmatory.py   text features, agent fingerprint, LLM auditor (--llm, --truncate-chars)
precision_auditor.py     auditor precision at 1–50% base rates, false alarms per catch
mechanism_analysis.py    exploratory: what predicts persistence or backfire
human_validation.py      export a blind labelling sheet, label in the terminal, judge the same sheet, score
label_messages.py        standalone labelling tool for extra raters (standard library only)
hpc_jobs/                Slurm job scripts used on the Plaksha HPC (cross-model auditor, judge sheet, ...)
mock_llm_client.py       offline mock model for testing pipelines without a GPU or API

Earlier stages (organic baseline, Step 6 mechanisms pilot) live in src/batch_organic.py, src/analyze_metrics.py,
run_step6.py, analyze_step6.py and related scripts; see "Project history".
```

`runs/` (manifests) and `logs/` (full trial transcripts) are generated data and are not committed.

---

## Setup

```bash
pip install openai numpy scipy scikit-learn
```

**Option A: NVIDIA API**

```powershell
$env:MODEL_PROVIDER="nvidia"
$env:NVIDIA_API_KEY="your_key"            # https://build.nvidia.com
# default model: nvidia/nemotron-3.5-lightning-30b-a3b
# gpt-oss runs also set: NVIDIA_MODEL=openai/gpt-oss-20b, LLM_EXTRA_BODY=none, LLM_MIN_TOKENS=1500
```

**Option B: self-hosted with vLLM (what most runs used)**

Serve the model with vLLM, then point the client at it:

```bash
vllm serve <model> --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 64   # + per-model flags below
export MODEL_PROVIDER=nvidia NVIDIA_API_KEY=local NVIDIA_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MAX_RPM=100000 LLM_MAX_CONCURRENCY=8 RUN_NS=<namespace>
```

Per-model flags used on a 48 GB RTX PRO 6000 slice: Nemotron `--trust-remote-code --quantization fp8`; gpt-oss none (native MXFP4); Gemma `--quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384`.

**Environment variables the code reads.** `RUN_NS` puts all manifests and logs under `runs/<ns>/` and `logs/<ns>/`. The 32-agent runs also use `PERSONAS_PATH`, `SCALE_ATTACKER_POOL` and `SCALE_ARMS`. Always start a fresh shell between experiments: a leftover variable once routed a run into the wrong configuration, and both runners now refuse to start when scale variables are set without a namespace.

---

## Reproducing the experiments

```bash
# 8-agent groups, one topic, organic twin + 4-of-8 attackers, stealth controls on
RUN_NS=hpc python run_confirmatory.py --topic free_speech_harm --seeds $(seq 1 20) \
    --arms organic attack4 --match-length --enforce-direction

# 32-agent groups
RUN_NS=scale32_hpc PERSONAS_PATH=config/personas_scale32.json \
SCALE_ATTACKER_POOL=<all 32 ids except A7_STATUS> SCALE_ARMS='{"attack_low": 4, "attack_high": 8}' \
    python run_confirmatory.py --topic free_speech_harm --seeds $(seq 1 10) --match-length --enforce-direction

# 512-agent network (offline test first: --mock --n-variants 2)
RUN_NS=society_mem_hpc python run_society.py --topic free_speech_harm --memory matched
RUN_NS=society_mem_hpc python run_society.py --topic free_speech_harm --memory matched --seeds 2 3 --arms organic attack10pct
```

Every runner writes a manifest and skips finished trials, so an interrupted or preempted job resumes where it stopped.

## Reproducing the analysis

```bash
python analyze_confirmatory.py --manifest runs/hpc/confirmatory_short_manifest__free_speech_harm.json --arms organic attack4
python pool_confirmatory.py                      # writes results_hpc2/pooled_cells.csv and pooled.json
python analyze_society.py --ns society_mem_hpc --quiet

# detection: full-length auditor, length-controlled auditor (own cache required)
M=runs/hpc/confirmatory_short_manifest__free_speech_harm.json
python detectability_confirmatory.py --long-manifest $M --short-manifest $M --llm --cache runs/hpc/llmcache_full__free_speech_harm.json
python detectability_confirmatory.py --long-manifest $M --short-manifest $M --llm --truncate-chars 120 --cache runs/hpc/llmcache_trunc120__free_speech_harm.json
python precision_auditor.py
python mechanism_analysis.py

# human validation
python human_validation.py export               # blind sheet + answer key
python label_messages.py sheet_<name>.csv      # each rater, on their own copy
python human_validation.py judge --tag nemotron  # judge labels the same sheet
python human_validation.py score --sheets sheet_a.csv sheet_b.csv results_hpc2/human_validation/sheet_judge_nemotron.csv
```

---

## Validity checks

| Concern | Check |
|---|---|
| Groups drift on their own | Every effect is attacked minus a same-seed twin with frozen pre-stances |
| Attackers may not argue their side | In-loop direction judge; blind human labels (93% vs 26%), κ = 0.63 with the judge |
| Auditor may just detect length | Messages cut to 120 characters; detection unchanged |
| Auditor tested only on its own model | Full 3 × 3 auditor × data matrix |
| Balanced tests flatter auditors | Precision reported at 1%, 5%, 10% and 50% base rates |
| Backend differences | API vs self-hosted calibration for Nemotron (effect within 0.01); every model kept on one backend; all gpt-oss runs self-hosted |
| One society per topic | Three independent networks per topic for the 10% arm, reported seed by seed |

## Limitations

- LLM agents, not people; effect sizes likely overstate human susceptibility.
- Prompted attackers over 6 rounds; no adaptive or fine-tuned attackers, no long horizons.
- Nemotron's college value and remote work cells ran on the NVIDIA API, the rest self-hosted.
- The 2% society arm has one network per topic; network-distance analyses are exploratory.
- One human rater so far; more raters in progress.

## Ethics

This work describes a working method for covert influence of AI-agent groups. We study it to measure the risk and to test defences: the detection tools are released alongside the attack setup, and all experiments use simulated agents only, with no real users or platforms.

---

## Project history

1. **Organic baseline** (8 topics, 63 trials): built the simulator and stance measurement; reduced stance-reading noise about 3× (lower temperature, averaged readings).
2. **Step 6 pilot** (one attacker, three mechanisms: disinformation, manufactured consensus, personalized persuasion): underpowered and later found to have a prompt bug (attackers were never told which side their target meant). Frozen as the "unspecified direction" pilot.
3. **Confirmatory study**: fixed direction, dose (1 and 4 of 8), paired twins, persistence phase, length matching and direction enforcement, two models on AI regulation.
4. **Extension**: 8 topics, a third model, 32-agent groups, self-hosting on HPC (one 40-trial topic in about 0.6 h instead of about 39 h over the API).
5. **Final**: 512-agent network, pooled meta-analysis, length-controlled and cross-model auditing, base-rate precision, mechanism analysis, human validation.

Backends: started on Groq (gpt-oss-20b, free tier too small), moved to the NVIDIA API (Nemotron), then self-hosted all three models with vLLM on the Plaksha University HPC.

## Acknowledgements

Experiments were run on the Plaksha University HPC cluster.

## License

MIT, see `LICENSE`.
