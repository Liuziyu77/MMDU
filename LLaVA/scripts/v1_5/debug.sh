#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=2
export MASTER_PORT=29501
export CPUS_PER_TASK=32
export QUOTA=reserved

export DATA_PATH=/mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/v1_5/finetune_json/finetune_data_mixed_llava_66k_mmdu_22k.json
export CKPT_PATH=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/pretrain_checkpoint
export VIT_PATH=/mnt/hwfile/mllm/liuziyu/CLIP_models/clip-vit-large-patch14-336
export SAVE_PATH=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/llava-v1.5-7b-lora-llava66k-mmdu22k-16k
export LEARNIG_RATE=2e-4

SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node 1 /mnt/petrelfs/liuziyu/V3Det/LLaVA/llava/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed /mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/zero3.json \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder /mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/v1_5/data \
    --vision_tower ${VIT_PATH} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ${SAVE_PATH} \
    --freeze_backbone True 