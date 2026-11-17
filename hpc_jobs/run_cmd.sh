#!/bin/bash
#SBATCH --job-name=run-model
#SBATCH --partition=rtx
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --exclude=cn03
set -euo pipefail
MODEL="$1"; MODE="$2"; NS="$3"; shift 3
case "$MODEL" in
  nemotron) HFID=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; SERVED=nvidia/nemotron-3.5-lightning-30b-a3b; EXTRA='--trust-remote-code --quantization fp8' ;;
  gptoss)   HFID=openai/gpt-oss-20b; SERVED=openai/gpt-oss-20b; EXTRA='' ;;
  gemma)    HFID=google/gemma-4-31B-it; SERVED=google/gemma-4-31b-it; EXTRA='--quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384 --limit-mm-per-prompt {"image":0,"audio":0}' ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
export HF_HOME=/scratch/$USER/hf HF_HUB_OFFLINE=1
source /scratch/$USER/envs/vllm/bin/activate
PORT=$((20000 + SLURM_JOB_ID % 10000)); while (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; do PORT=$((PORT+1)); done
vllm serve $HFID --served-model-name $SERVED --port $PORT --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 $EXTRA > vllm-server-$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "SERVER DID NOT START"; tail -40 vllm-server-$SLURM_JOB_ID.log; exit 1; }
echo "server up after ~$((i*10))s | model=$MODEL mode=$MODE ns=$NS | args: $*"
unset RUN_NS PERSONAS_PATH SCALE_ATTACKER_POOL SCALE_ARMS NVIDIA_MODEL LLM_EXTRA_BODY LLM_MIN_TOKENS GROQ_API_KEY GROQ_MODEL
export MODEL_PROVIDER=nvidia NVIDIA_API_KEY=local NVIDIA_BASE_URL=http://127.0.0.1:$PORT/v1 \
       LLM_MAX_RPM=100000 LLM_MAX_CONCURRENCY=8 RUN_NS="$NS"
if [ "$MODEL" = "gptoss" ]; then export NVIDIA_MODEL=openai/gpt-oss-20b LLM_EXTRA_BODY=none LLM_MIN_TOKENS=1500; fi
if [ "$MODEL" = "gemma" ]; then export NVIDIA_MODEL=google/gemma-4-31b-it; fi
if [ "$MODE" = "scale" ]; then
  export PERSONAS_PATH=config/personas_scale32.json \
         SCALE_ATTACKER_POOL="A1,A2,A3,A4,A5,A6,A8_ADVERSARY$(printf ',A%d' $(seq 9 32))" \
         SCALE_ARMS='{"attack_low": 4, "attack_high": 8}'
fi
env | grep -E '^(RUN_NS|PERSONAS_PATH|SCALE_|NVIDIA_|MODEL_PROVIDER|LLM_)' | grep -v API_KEY | sort
cd ~/stealthy_influence
bash -c "$CMD"
kill $SERVER
echo JOB_DONE
