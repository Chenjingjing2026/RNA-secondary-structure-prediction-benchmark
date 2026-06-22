#!/bin/bash

# 后台运行训练脚本并保存日志
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --device cuda \
    --diffusion_dim 8 \
    --diffusion_steps 20 \
    --cond_dim 8 \
    --dataset tRNA \
    --batch_size 1 \
    --dp_rate 0.1 \
    --lr 0.0001 \
    --num_workers 6 \
    --warmup 5 \
    --seed 2023 \
    --log_wandb False \
    --epochs 100 \
    --eval_every 4 \
    --u_conditioner_ckpt ufold_train_tRNA_short.pt \
    > /home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/crossfamily_short/tRNA/training1.log 2>&1 &

echo "训练进程已在后台启动，PID: $!"