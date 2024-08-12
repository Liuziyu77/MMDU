#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=1
export NNODES=1
export MASTER_PORT=29507
export CPUS_PER_TASK=32
export QUOTA=reserved

export DATA_PATH=/mnt/petrelfs/liuziyu/RLHF/make_data/mantis/data/sampled_mantis_llava665k_multi_45k.json
export CKPT_PATH=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/llava-v1.5-7b
export VIT_PATH=/mnt/hwfile/mllm/liuziyu/CLIP_models/clip-vit-large-patch14-336
export SAVE_PATH=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/lora_RLHF_llava_sft_sampled_mantis_llava665k_multi_45k
export LEARNIG_RATE=2e-4

SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

torchrun --nnodes 1 --nproc_per_node 1 --master_port 29507 /mnt/petrelfs/liuziyu/V3Det/LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/zero2.json \
    --model_name_or_path /mnt/hwfile/mllm/liuziyu/finetune_LLaVa/llava-v1.5-7b \
    --version v1 \
    --data_path /mnt/petrelfs/liuziyu/RLHF/make_data/mantis/data/sampled_mantis_llava665k_multi_45k.json \
    --image_folder /mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/v1_5/data/cl_data \
    --vision_tower /mnt/hwfile/mllm/liuziyu/CLIP_models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /mnt/hwfile/mllm/liuziyu/finetune_LLaVa/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --freeze_backbone True
