#!/bin/bash
# Cross-model LLM auditor: the model served in this job (RUN_NS=xaud_<auditor>) audits the OTHER two models'
# conversations, full-length messages, 8 topics each. Own-model results already exist, so this fills the off-diagonal
# of the auditor x data matrix. Each (auditor, data, topic) gets its own cache, so nothing is reused across cells.
cd ~/stealthy_influence
AUD=${RUN_NS#xaud_}
man() {  # data-model topic -> manifest
  local d=$1 t=$2 base
  case $d in
    nemotron) if [ $t = college_value ] || [ $t = remote_work ]; then echo runs/confirmatory_short_manifest__$t.json; return; fi
              base=runs/hpc ;;
    gptoss)   base=runs/gptoss_hpc ;;
    gemma)    base=runs/gemma_hpc ;;
  esac
  if [ $t = ai_regulation ]; then echo $base/confirmatory_short_manifest.json; else echo $base/confirmatory_short_manifest__$t.json; fi
}
mkdir -p runs/xaud/out
for D in nemotron gptoss gemma; do
  [ $D = $AUD ] && continue
  for T in ai_regulation college_value free_speech_harm remote_work traditional_family_roles religion_public_life centralized_governance individual_vs_collective; do
    M=$(man $D $T)
    python -u detectability_confirmatory.py --long-manifest $M --short-manifest $M --llm \
      --cache runs/xaud/${AUD}_on_${D}__$T.json --out runs/xaud/out/${AUD}_on_${D}_$T.json \
      > results_hpc2/xaud_${AUD}_on_${D}_$T.txt 2>&1
    echo "done $AUD on $D $T"
  done
done
