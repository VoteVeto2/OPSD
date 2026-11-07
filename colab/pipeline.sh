#!/bin/bash
# Colab A100 pipeline: train OPSD Qwen3-1.7B to step 100, then AIME24 Avg@12 eval
# of base + checkpoints 25/50/75/100. Each result lands in $RES/ as soon as it is done.
# Usage (on VM): nohup bash colab/pipeline.sh > /content/pipeline.log 2>&1 &
set -u
cd /content/OPSD
export UV_FROZEN=1 UV_LINK_MODE=copy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-offline} TRL_EXPERIMENTAL_SILENCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
RUN=/content/opsd_runs/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005
RES=/content/results; mkdir -p $RES
STOP_STEP=${STOP_STEP:-100}

stage() { echo "[pipeline] $(date -u +%FT%TZ) $*"; }

if [ ! -f $RUN/checkpoint-$STOP_STEP/adapter_model.safetensors ]; then
  stage "train start"
  bash scripts/run_opsd_1b_1gpu.sh > /content/train.log 2>&1 &
  TPID=$!
  # Linear LR schedule spans the full 30-epoch run (as in the paper); stop after step 100 is saved.
  while kill -0 $TPID 2>/dev/null; do
    if [ -f $RUN/checkpoint-$STOP_STEP/adapter_model.safetensors ] && [ -f $RUN/checkpoint-$STOP_STEP/trainer_state.json ]; then
      sleep 20; stage "checkpoint-$STOP_STEP saved, stopping training"
      pkill -TERM -P $TPID; kill -TERM $TPID; sleep 15; pkill -9 -f opsd_train.py; break
    fi
    sleep 15
  done
  wait $TPID 2>/dev/null
  [ -f $RUN/checkpoint-$STOP_STEP/adapter_model.safetensors ] || { stage "train FAILED"; exit 1; }
  stage "train done"
fi
sleep 10

cd eval
for step in 0 25 50 75 100; do
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
stage "ALL DONE"
