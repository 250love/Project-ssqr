#!/usr/bin/env bash
set -e

accelerate launch -m src.train \
  --stage sft \
  --model_name_or_path /root/autodl-tmp/SSQR_Stage_2/model2code \
  --dataset ssqr \
  --dataset_dir /root/autodl-tmp/SSQR_Stage_2/LLaMA-Factory/data \
  --finetuning_type lora \
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
  --output_dir /root/autodl-tmp/SSQR_Stage_2/runs/kgcodes-llama2-7b-lora \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --val_size 0.05 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 2 \
  --gradient_checkpointing \
  --flash_attn auto \
  --cutoff_len 2048 \
  --report_to none \
  --packing False \
  --bf16 \
  --quantization_method bnb \
  --quantization_bit 4 \
  --quantization_type nf4 \
  --double_quantization
