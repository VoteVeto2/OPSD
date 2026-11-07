#!/bin/bash
# Qwen3-1.7B OPSD on a single GPU (default: Colab A100 80GB, `colab new --gpu A100 --high-mem`).
# Same recipe as run_opsd_1b.sh; effective batch kept at 32
# (orig: 4 GPUs x 4 x GA2 -> here: 1 GPU x 4 x GA8, same per-GPU micro-batch as the paper).
# vLLM memory share 0.6 -> 0.35: 1.7B rollouts need little KV cache, and the full-vocab JSD
# needs the room (each [4,1024,151936] fp32 tensor is 2.3 GiB). Sampling is unaffected.
# On a 40GB A100 use: PER_DEVICE_BS=2 GA=16 VLLM_MEM=0.25 (micro-batch 4 OOMs there).
OUTPUT_DIR=${OUTPUT_DIR:-/content/opsd_runs}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
VLLM_MEM=${VLLM_MEM:-0.35}
PER_DEVICE_BS=${PER_DEVICE_BS:-4}
GA=${GA:-8}

uv run accelerate launch \
    --config_file accelerate_1gpu.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps $GA \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path $MODEL \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_checkpointing \
    --gradient_accumulation_steps $GA \
    --output_dir $OUTPUT_DIR \
    --run_config qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005 \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization $VLLM_MEM \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
