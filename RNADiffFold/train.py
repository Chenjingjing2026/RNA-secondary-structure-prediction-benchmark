# -*- coding: utf-8 -*-
import torch
import argparse
import collections
import os
from common.utils import add_parent_path, set_seeds

add_parent_path(level=1)

from models.model import get_model, get_model_id, add_model_args
from optim.multistep import get_optim, get_optim_id, add_optim_args
from datasets.data import get_data_id, add_data_args, get_data
from datasets.data_gt import get_data_id as get_data_id_gt,add_data_args as  add_data_args_gt,get_data as get_data_gt
from experiment import Experiment, add_exp_args

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

#
# if Reload:
#     def checkpoint_load(check_path, name='checkpoint.pt'):
#         checkpoint = torch.load(os.path.join(check_path, name))
#         current_epoch = checkpoint['current_epoch']
#         train_metrics = checkpoint['train_metrics']
#         eval_metrics = checkpoint['eval_metrics']
#         eval_epochs = checkpoint['eval_epochs']
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         if scheduler_iter: scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
#         if scheduler_epoch: scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])
#
#         return current_epoch,train_metrics,eval_metrics,eval_epochs,model,optimizer,scheduler_iter,scheduler_epoch
#
#
#     check_path='/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/crossfamily/grp2/logs/rnadifffold/grp2_160_multinomial_diffusion_adam/2025-05-23_17-12-04/check/'
#     name='best_checkpoint_32.pt'
#     current_epoch,train_metrics,eval_metrics,eval_epochs,model,optimizer,scheduler_iter,scheduler_epoch=checkpoint_load(check_path, name)
#     exp = Experiment(args=args,
#                      data_id=data_id,
#                      model_id=model_id,
#                      optim_id=optim_id,
#                      train_loader=train_loader,
#                      val_loader=val_loader,
#                      test_loader=test_loader,
#                      model=model,
#                      optimizer=optimizer,
#                      scheduler_iter=scheduler_iter,
#                      scheduler_epoch=scheduler_epoch,
#                      current_epoch=current_epoch
#                      )

Reload=False
if Reload:
    def checkpoint_load(check_path, name='checkpoint.pt', device='cuda'):
        # 加载检查点并自动映射到指定设备
        checkpoint = torch.load(os.path.join(check_path, name), map_location=device)

        current_epoch = checkpoint['current_epoch']
        train_metrics = checkpoint['train_metrics']
        eval_metrics = checkpoint['eval_metrics']
        eval_epochs = checkpoint['eval_epochs']

        # 加载模型（自动匹配设备）
        model.load_state_dict(checkpoint['model'])

        # 加载优化器并转移状态到GPU
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param in model.parameters():
            param.data = param.data.to(device)  # 确保模型参数在GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)  # 转移优化器状态

        # 加载调度器
        if scheduler_iter and 'scheduler_iter' in checkpoint:
            scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if scheduler_epoch and 'scheduler_epoch' in checkpoint:
            scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

        return current_epoch, train_metrics, eval_metrics, eval_epochs

    # 使用示例
    check_path = '/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/crossfamily/grp2/logs/rnadifffold/grp2_160_multinomial_diffusion_adam/2025-06-17_13-16-29/check/'
    name = 'best_checkpoint_24.pt'

    # 确保模型先在GPU上
    model = model.to('cuda')

    # 加载检查点（自动处理设备转移）
    current_epoch, train_metrics, eval_metrics, eval_epochs = checkpoint_load(
        check_path=check_path,
        name=name,
        device='cuda'  # 明确指定目标设备
    )

    # 初始化Experiment
    exp = Experiment(
        args=args,
        data_id=data_id,
        model_id=model_id,
        optim_id=optim_id,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,  # 已在GPU上
        optimizer=optimizer,  # 状态已转移
        scheduler_iter=scheduler_iter,
        scheduler_epoch=scheduler_epoch,
        current_epoch=current_epoch  # 建议参数名用start_epoch更明确
    )
else:
# training
    current_epoch=0
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

exp.run()
