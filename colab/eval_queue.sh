#!/bin/bash
# Run AIME24 Avg@12 evals for the given steps, one after another, on this VM.
# Waits for any eval already running to finish first. 0 = base model.
# Usage (on VM): nohup bash colab/eval_queue.sh 100 25 > /dev/null 2>&1 &
set -u
cd /content/OPSD/eval
export UV_FROZEN=1 UV_LINK_MODE=copy TRL_EXPERIMENTAL_SILENCE=1
RUN=${RUN:-/content/opsd_runs/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005}
RES=/content/results; mkdir -p $RES
stage() { echo "[pipeline] $(date -u +%FT%TZ) $*" >> /content/pipeline.log; }
while pgrep -f "python evaluate_math.py" > /dev/null; do sleep 30; done
for step in "$@"; do
  out=$RES/aime24_step$step.json
  [ -f $out ] && { stage "skip step $step (exists)"; continue; }
  ck=(); [ $step -gt 0 ] && ck=(--checkpoint_dir $RUN/checkpoint-$step)
  stage "eval step $step start"
  uv run python evaluate_math.py --base_model Qwen/Qwen3-1.7B "${ck[@]}" \
      --dataset aime24 --val_n 12 --temperature 1.0 --tensor_parallel_size 1 \
      --no_enforce_eager --output_file $out > /content/eval_step$step.log 2>&1 \
    && stage "eval step $step done: $(grep -a 'Final Average' /content/eval_step$step.log)" \
    || stage "eval step $step FAILED"
done
stage "queue done: $*"
