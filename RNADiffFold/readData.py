# -*- coding: utf-8 -*-
import torch
import argparse
import collections

from common.utils import add_parent_path, set_seeds

add_parent_path(level=1)

from models.model import get_model, get_model_id, add_model_args
from optim.multistep import get_optim, get_optim_id, add_optim_args
# from datasets.data import get_data_id, add_data_args, get_data
from datasets.data_gt import get_data_id, add_data_args, get_data
from experiment import Experiment, add_exp_args


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


# data
data_id = get_data_id(args)
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')
train_loader, val_loader, test_loader = get_data(args, alphabet)
print('train_loader len:',len(train_loader.dataset) )
# 假设 train_batch 是一个普通的 tuple
train_batch = next(iter(train_loader))
print('train_batch type:',type(train_batch))
print('train_batch keys:',train_batch.keys())
# 将 tuple 转换成 RNA_SS_data 的命名元组格式
train_batch_data = RNA_SS_data(contact=train_batch['contact'],
                                data_fcn_2=train_batch['data_fcn_2'],
                                seq_raw=train_batch['data_seq_raw'],
                                length=train_batch['data_length'],
                                name=train_batch['data_name'])

# 打印 RNA_SS_data 格式的内容
print("=== Train Loader ===")
print(f"contact shape: {train_batch_data.contact.shape}")        # 接触矩阵
print(f"data_fcn_2 shape: {train_batch_data.data_fcn_2.shape}")  # 特征矩阵
print(f"seq_raw shape: {train_batch_data.seq_raw.shape}")
print(f"seq_raw shape: {train_batch_data.seq_raw}")  # 原始序列