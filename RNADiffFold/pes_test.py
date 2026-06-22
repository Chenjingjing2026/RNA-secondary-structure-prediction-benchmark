# -*- coding: utf-8 -*-
import torch
import argparse
import collections

from common.utils import add_parent_path, set_seeds

add_parent_path(level=1)

from models.model import get_model, get_model_id, add_model_args
from optim.multistep import get_optim, get_optim_id, add_optim_args
from datasets.data import get_data_id, add_data_args, get_data
from datasets.data_gt import get_data_id as get_data_id_gt,add_data_args as  add_data_args_gt,get_data as get_data_gt
from pes_experiment import pes_Experiment, add_exp_args

from experiment import Experiment

GT=True
# Setup
parser = argparse.ArgumentParser()

add_model_args(parser)
add_data_args(parser)
add_optim_args(parser)
add_exp_args(parser)

args = parser.parse_args()

set_seeds(args.seed)

# model
model_id = get_model_id(args)
model, alphabet = get_model(args)
# model_ckpt=torch.load('/local4/fyf/TEST/RNADiffFold-master/ckpt_19.pth',map_location='cpu')
# model.load_state_dict(model_ckpt)

# ckpt_path = '..'
# checkpoint = torch.load(ckpt_path, map_location='cpu')
# model.load_state_dict(checkpoint['model'])
# print('load model from {}'.format(ckpt_path))

# data
if GT:
    data_id = get_data_id_gt(args)
else:
    data_id = get_data_id(args)
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')

if GT:
    train_loader, val_loader, test_loader = get_data_gt(args, alphabet)
else:
    train_loader, val_loader, test_loader = get_data(args, alphabet)

# optimizer
optim_id = get_optim_id(args)
optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)


# 加载预训练模型
def load_model(model, model_path, device):
    try:
        # 加载整个checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # 检查checkpoint中包含哪些键
        print("Checkpoint keys:", checkpoint.keys())

        # 根据不同格式提取模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            # 如果保存的是整个实验状态
            model.load_state_dict(checkpoint['model'])
        else:
            # 尝试直接加载（可能是纯模型权重）
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"✓ Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


# 使用方式
model_path = "/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/pes1/logs/rnadifffold/pes_160_multinomial_diffusion_adam/2025-09-09_12-04-07/check/best_checkpoint_60.pt"
model = load_model(model, model_path, args.device)
print('success')

current_epoch=0
# training
# exp = pes_Experiment(args=args,
#                  data_id=data_id,
#                  model_id=model_id,
#                  optim_id=optim_id,
#                  train_loader=train_loader,
#                  val_loader=val_loader,
#                  test_loader=test_loader,
#                  model=model,
#                  optimizer=optimizer,
#                  scheduler_iter=scheduler_iter,
#                  scheduler_epoch=scheduler_epoch,
#                  current_epoch=current_epoch
#                  )
#
# exp.test_fn()

exp = Experiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 test_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch,
                 current_epoch=current_epoch
                 )

exp.test_fn()