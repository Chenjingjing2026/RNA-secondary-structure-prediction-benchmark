#!/bin/bash

# 后台运行训练脚本并保存日志
python readData.py \
    --device cuda:0 \
    --diffusion_dim 8 \
    --diffusion_steps 20 \
    --cond_dim 8 \
    --dataset all \
    --batch_size 3 \
    --dp_rate 0.1 \
    --lr 0.0001 \
    --num_workers 6 \
    --warmup 5 \
    --seed 2023 \
    --log_wandb False \
    --epochs 100 \
    --eval_every 4 \
    --u_conditioner_ckpt ufold_train_alldata.pt
