#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys

import torch as t
import argparse
# import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib

matplotlib.use('Agg')
# import time
from tqdm import tqdm

t.manual_seed(1080)
import numpy as np

np.random.seed(1080)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False

from model import *
from loader import *
from evaluate import *
# from torchsummary import summary

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
######################
###  Get args
######################
import torch
print(torch.__version__)
print(torch.cuda.is_available())

parser = argparse.ArgumentParser(
    description='This is the pytorch implementation of SPOT-RNA (https://doi.org/10.1038/s41467-019-13395-9)',
    add_help=True)

## Input and Output
parser.add_argument('--datadir', dest='datadir', type=str, nargs='?', required=True,
                    help='Diretory of data, must include TR0, VL0 and TS0 subdirs')
parser.add_argument('--sessName', dest='sessName', type=str, nargs='?', required=True,
                    help='Session name, the model and figures will be saved with this name')

## Residual Blocks
parser.add_argument('--resBlockNum', dest='resBlockNum', type=int, nargs='?', default=12,
                    help='Number of residual blocks, (N_A) in paper (default: 12)')
parser.add_argument('--chanNum', dest='chanNum', type=int, nargs='?', default=48,
                    help='Number of channels in residual blocks, (D_RES) in paper (default: 48)')
parser.add_argument('--dilaFactor', dest='dilaFactor', type=int, nargs='?', default=1,
                    help='Dilation factor for convolution layer (default: 1)')

## Bi-directional LSTM
parser.add_argument('--disableLSTM', dest='disableLSTM', action='store_true',
                    help='Disable bi-directional LSTM layer')
parser.add_argument('--LSTMHiddenDim', dest='LSTMHiddenDim', type=int, nargs='?', default=200,
                    help='Hidden dimension of Bi-LSTM, (D_BL) in paper (default: 200)')

## Fully connected layers
parser.add_argument('--FCHiddenDim', dest='FCHiddenDim', type=int, nargs='?', default=512,
                    help='Dimension of fully connected hidden layers, (D_FC) in paper (default: 512)')
parser.add_argument('--FCHiddenLayer', dest='FCHiddenLayer', type=int, nargs='?', default=2,
                    help='Number of fully connected hidden layers, (N_B) in paper (default: 2)')

## Loss function
parser.add_argument('--lossAlpha', dest='lossAlpha', type=int, nargs='?', default=4,
                    help='The weight of positive samples. The weight ratio of positive and negative samples is (n + 1): 1 (default: 4)')

## Other configures
parser.add_argument('--epoch', dest='epoch', type=int, nargs='?', default=100,
                    help='Number of epochs (default: 100)')
parser.add_argument('--device', dest='device', type=str, nargs='?', default='cuda',
                    help='cpu or cuda device (default: cpu)')

# parser.add_argument('--filelist', dest='filelist', type=str, nargs='?', required=True,
#                     help='filelist name')


args = parser.parse_args()

if args.dilaFactor < 1:
    print("Error: --dilaFactor should larger than 0")
    exit(-1)
if args.chanNum < 1:
    print("Error: --chanNum should larger than 0")
    exit(-1)

# device = t.device(args.device)
device = 'cuda'
######################
###  Build model
######################

seq_modules_list = [InputConvUnit(out_dim=args.chanNum)]
for i in range(args.resBlockNum):
    seq_modules_list.append(ResBlock(dim=args.chanNum, dilation=args.dilaFactor))

if not args.disableLSTM:
    seq_modules_list.append(BiLSTM(in_dim=args.chanNum, hidden_dim=args.LSTMHiddenDim, dropout=0))
    seq_modules_list.append(BooleanMask(dim=2 * args.LSTMHiddenDim))
    seq_modules_list.append(
        FCL(in_dim=2 * args.LSTMHiddenDim, hidden_layers=args.FCHiddenLayer, hidden_dim=args.FCHiddenDim))

else:
    seq_modules_list.append(BooleanMask(dim=args.chanNum))
    seq_modules_list.append(FCL(in_dim=args.chanNum, hidden_layers=args.FCHiddenLayer, hidden_dim=args.FCHiddenDim))

model = nn.Sequential(*seq_modules_list)

# 检查是否有多个 GPU 可用
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # 将模型包装为 DataParallel

# 将模型移动到设备
model.to(device)
# state_dict = torch.load('model' + str(0) + '.model')
# model.load_state_dict(state_dict)

# model.cpu()
# summary(model, (8, 67, 67), batch_size=1, device='cpu')
# print(model)

######################
###  Optimizer
######################

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

######################
###  Load data
######################
DATASET='telomerase'
if DATASET=='shuffle_long':
    train_file ='train.txt'
    val_file = 'valid.txt'
    test_file = 'test_long.txt'
elif DATASET=='shuffle_medium':
    train_file ='train.txt'
    val_file = 'valid.txt'
    test_file = 'test_medium.txt'
elif DATASET=='shuffle_long':
    train_file ='train.txt'
    val_file = 'valid.txt'
    test_file = 'test_long.txt'
elif DATASET=='telomerase':
    train_file ='train.txt'
    val_file = 'valid.txt'
    test_file = 'test.txt'
Train = False
if Train:
    print("Start to load training data...")
    training_loader = SPOTDataSetDot(os.path.join(args.datadir, "train"), device=device,flist=train_file)
    print('Loaded Training dataset, Len: ',len(training_loader))
    print("Start to load validating data...")
    validation_loader = SPOTDataSetDot(os.path.join(args.datadir, "valid"), device=device,flist=val_file)
    print('Loaded validation dataset, Len: ',len(validation_loader))
    
    ######################
    ###  Train model
    ######################
    
    model.train()
    hist_loss = []
    hist_acc = []
    hist_preci = []
    hist_recall = []
    hist_mcc = []
    report_num = 1000
    
    last_highest_validation_F1 = 0
    pbar = tqdm(total=report_num, leave=False, ncols=50)
    for epoch in range(args.epoch):
    
        loss_list = []
        acc_list = []
        preci_list = []
        recall_list = []
        MCC_list = []
    
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_train_preci = []
        epoch_train_recall = []
        epoch_train_MCC = []
    
        i = 0
        for (rna_name, seq_matrix), dot_matrix in training_loader:
            try:
                if seq_matrix.shape[2] > 1024:
                    #print(rna_name,' len is',seq_matrix.shape[2],' > 1024,will skill')
                    continue
                print('Start Train:',rna_name)
                pred = model(seq_matrix)
                preci, recall, acc, mcc = precision_recall_accuracy(t.sigmoid(pred.data), dot_matrix)
    
                loss = LogisticLoss(pred, dot_matrix)
                # print(seq_matrix.shape)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                acc_list.append(acc)
                preci_list.append(preci)
                recall_list.append(recall)
                MCC_list.append(mcc)
    
                epoch_train_loss.append(loss.item())
                epoch_train_acc.append(acc)
                epoch_train_preci.append(preci)
                epoch_train_recall.append(recall)
                epoch_train_MCC.append(mcc)
    
                del loss, pred, seq_matrix, dot_matrix
                t.cuda.empty_cache()
    
                pbar.update(1)
                i += 1
                if i % report_num == 0:
                    pbar.close()
                    # with tf.Session() as sess:
                    #     # 运行会话以获取张量的值
                    #     mcc_values = sess.run(MCC_list)
                    print("sample [%s|%s]; epoch [%s|%s]; loss=%.5f, acc=%.5f, preci=%.5f, recall=%.5f, MCC=%.5f" % (
                        i, len(training_loader), epoch, args.epoch,
                        np.mean(loss_list), np.mean(acc_list), np.mean(preci_list), np.mean(recall_list), np.mean(mcc_values)))
                    loss_list = []
                    acc_list = []
                    pbar = tqdm(total=report_num, leave=False, ncols=50)
            except Exception as e:
                # print(rna_name, len(seq_matrix))
                continue
        pbar.close()
        
        print('Start Valid')
        val_acc, val_loss, val_preci, val_recall, val_mcc = evaluate_model(model, validation_loader)
        val_F1 = 2 * (val_preci * val_recall) / (val_preci + val_recall)
    
        hist_loss.append([np.mean(epoch_train_loss), val_loss])
        hist_acc.append([np.mean(epoch_train_acc), val_acc])
        hist_preci.append([np.mean(epoch_train_preci), val_preci])
        hist_recall.append([np.mean(epoch_train_recall), val_recall])
        hist_mcc.append([np.mean(epoch_train_MCC), val_mcc])
    
    
        print(
            "=====> epoch [%s|%s]; loss=%.5f, acc=%.5f, val_acc=%.5f, val_loss=%.5f, val_preci=%.5f, val_recall=%.5f, val_F1=%.5f, val_MCC=%.5f" % (
                epoch,
                args.epoch, np.mean(epoch_train_loss), np.mean(epoch_train_acc), val_acc, val_loss, val_preci, val_recall,
                val_F1, val_mcc))
    
        if len(hist_loss) >= 3:
            plot_acc_loss(args.sessName, hist_acc, hist_loss, hist_preci, hist_recall, hist_mcc)
    
        if val_F1 > last_highest_validation_F1:
            last_highest_validation_F1 = val_F1
            t.save(model.state_dict(), args.sessName + ".model")
else:
    print('Loading Model')
    model.load_state_dict(torch.load(args.sessName + ".model"))
    
######################
###  Test model
######################

print("Start to load testing data...")
test_loader = SPOTDataSetDot(os.path.join(args.datadir, "test"), device=device, flist=test_file)
print('Loaded Test Data success')
print('Start Evaluate')
test_acc, test_loss, test_preci, test_recall, test_mcc = evaluate_model(model, test_loader)
test_F1 = 2 * (test_preci * test_recall) / (test_preci + test_recall)
print("test_acc=%.5f, test_loss=%.5f, test_preci=%.5f, test_recall=%.5f, test_F1=%.5f, test_MCC=%.5f" % (
    test_acc, test_loss, test_preci, test_recall, test_F1, test_mcc))
