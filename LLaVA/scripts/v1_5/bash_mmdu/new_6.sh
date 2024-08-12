#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=8
export MASTER_PORT=29501
export CPUS_PER_TASK=32
export QUOTA=reserved

export DATA_PATH=/mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/v1_5/finetune_json/finetune_data_mixed_llava_166k_mmdu_22k.json
export CKPT_PATH=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/pretrain_checkpoint
export VIT_PATH=/mnt/hwfile/mllm/liuziyu/CLIP_models/clip-vit-large-patch14-336
export SAVE_PATH=/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/lora-llava166k-mmdu22k-16k
export LEARNIG_RATE=2e-4

SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p mllm \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA} \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} /mnt/petrelfs/liuziyu/V3Det/LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
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
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb'
