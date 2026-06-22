# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import torch as t
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib
import torch

matplotlib.use('Agg')
np.random.seed(1080)


def LogisticLoss(y_hat, y, lossAlpha=4):
    return ((1 - y) * y_hat + (lossAlpha * y + 1) * ((-y_hat).clamp(min=0) + (1 + (-y_hat.abs()).exp()).log())).mean()

#
# def precision_recall_accuracy(pred, target):
#     pred = (pred > 0.5).float()
#     pred_pos = pred.squeeze().nonzero().squeeze()
#     target_pos = target.squeeze().nonzero().squeeze()
#
#     truenum = t.sum(pred == target).cpu().item()
#     acc = truenum / target.shape[0]
#
#     if pred_pos.numel() == 0:
#         # return 0.0, 0.0, acc, 0.0
#         # 如果分母为零，MCC 设为 0
#         # mcc = tf.convert_to_tensor(0.0)
#         # return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(acc), torch.tensor(0.0)
#         return tf.convert_to_tensor(0.0), tf.convert_to_tensor(0.0), tf.convert_to_tensor(acc), tf.convert_to_tensor(0.0)
#
#     elif pred_pos.numel() == 1:
#         pred_set = set([pred_pos.cpu().tolist()])
#         target_set = set(target_pos.reshape([-1]).cpu().numpy())
#         precision = len(pred_set & target_set) / (len(pred_set) + 0.01)
#         recall = len(pred_set & target_set) / (len(target_set) + 0.01)
#
#         # 计算 TP, TN, FP, FN
#         TP = len(pred_set & target_set)
#         FP = len(pred_set - target_set)
#         FN = len(target_set - pred_set)
#         TN = target.shape[0] - TP - FP - FN  # 总样本数减去 TP, FP, FN
#
#     else:
#         pred_set = set(pred_pos.cpu().numpy())
#         target_set = set(target_pos.reshape([-1]).cpu().numpy())
#         precision = len(pred_set & target_set) / (len(pred_set) + 0.01)
#         recall = len(pred_set & target_set) / (len(target_set) + 0.01)
#
#         # 计算 TP, TN, FP, FN
#         TP = len(pred_set & target_set)
#         FP = len(pred_set - target_set)
#         FN = len(target_set - pred_set)
#         TN = target.shape[0] - TP - FP - FN  # 总样本数减去 TP, FP, FN
#
#         # 假设 TP, TN, FP, FN 在其他地方已经计算出来，确保它们是 Tensor 类型
#     # TP = torch.tensor(TP, dtype=torch.float32)
#     # TN = torch.tensor(TN, dtype=torch.float32)
#     # FP = torch.tensor(FP, dtype=torch.float32)
#     # FN = torch.tensor(FN, dtype=torch.float32)
#     # 将 TP, TN, FP, FN 转换为 TensorFlow 张量
#     TP = tf.convert_to_tensor(TP, dtype=tf.float32)
#     TN = tf.convert_to_tensor(TN, dtype=tf.float32)
#     FP = tf.convert_to_tensor(FP, dtype=tf.float32)
#     FN = tf.convert_to_tensor(FN, dtype=tf.float32)
#
#     # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
#     a = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
#         # print(type(a))
#         # if a == 0:
#         #     mcc = 0.0
#         # else:
#         #     mcc = (TP * TN - FP * FN) / torch.sqrt(a)
#         # Then perform the calculation
#         # mcc = (TP * TN - FP * FN) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
#         #
#         # print(f"MCC: {mcc}")
#     # print(type(a))
#     # 计算 MCC
#         # 检查输入是否为 TensorFlow 张量
#     if isinstance(TP, tf.Tensor) and isinstance(FP, tf.Tensor) and \
#             isinstance(TN, tf.Tensor) and isinstance(FN, tf.Tensor):
#         # 计算分母
#         denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
#         # 检查分母是否为零
#         if denominator != 0:
#             # 计算 MCC
#             mcc = (TP * TN - FP * FN) / tf.sqrt(denominator)
#         else:
#             # mcc = 0.0  # 防止除以零的情况
#             # mcc = torch.tensor(0.0)  # 防止除以零的情况
#             # 如果分母为零，MCC 设为 0
#             mcc = tf.convert_to_tensor(0.0)
#     else:
#         # 如果输入不是 TensorFlow 张量，尝试转换为张量
#         try:
#             TP_tensor = tf.convert_to_tensor(TP, dtype=tf.float32)
#             FP_tensor = tf.convert_to_tensor(FP, dtype=tf.float32)
#             TN_tensor = tf.convert_to_tensor(TN, dtype=tf.float32)
#             FN_tensor = tf.convert_to_tensor(FN, dtype=tf.float32)
#             # 计算分母
#             denominator = (TP_tensor + FP_tensor) * (TP_tensor + FN_tensor) * (TN_tensor + FP_tensor) * (TN_tensor + FN_tensor)
#             # 检查分母是否为零
#             if denominator != 0:
#                 # 计算 MCC
#                 mcc = (TP_tensor * TN_tensor - FP_tensor * FN_tensor) / tf.sqrt(denominator)
#             else:
#                 # mcc = 0.0  # 防止除以零的情况
#                 # mcc = torch.tensor(0.0)  # 防止除以零的情况
#                 # 如果分母为零，MCC 设为 0
#                 mcc = tf.convert_to_tensor(0.0)
#         except Exception as e:
#             print(f"Error converting inputs to TensorFlow tensors: {e}")
#             # mcc = 0.0
#             # mcc = torch.tensor(0.0)  # 防止除以零的情况
#             # 如果分母为零，MCC 设为 0
#             mcc = tf.convert_to_tensor(0.0)
#
#     return precision, recall, acc, mcc
#
#
# def evaluate_model(model, dataloader):
#     model.eval()
#     acc_list = []
#     mcc_list = []
#     loss_list = []
#     precision_list = []
#     recall_list = []
#     with t.set_grad_enabled(False):
#         for (rna_name, seq_matrix), dot_matrix in dataloader:
#             pred = model(seq_matrix)
#
#             precision, recall, acc, mcc = precision_recall_accuracy(t.sigmoid(pred), dot_matrix)
#             precision_list.append(precision)
#             recall_list.append(recall)
#             acc_list.append(acc)
#             mcc_list.append(mcc)
#
#             loss = LogisticLoss(pred, dot_matrix)
#             loss_list.append(loss.item())
#
#     model.train()
#     with tf.Session() as sess:
#         # 运行会话以获取张量的值
#         mcc_values = sess.run(mcc_list)
#     mcc_mean = np.mean(mcc_values)
#     return np.mean(acc_list), np.mean(loss_list), np.mean(precision_list), np.mean(recall_list),mcc_mean

def precision_recall_accuracy(pred, target):
    pred = (pred > 0.5).float()
    pred_pos = pred.squeeze().nonzero().squeeze()
    target_pos = target.squeeze().nonzero().squeeze()

    truenum = t.sum(pred == target).cpu().item()
    acc = truenum / target.shape[0]

    if pred_pos.numel() == 0:
        return 0.0, 0.0, acc, 0.0
    elif pred_pos.numel() == 1:
        pred_set = set([pred_pos.cpu().tolist()])
        target_set = set(target_pos.reshape([-1]).cpu().numpy())
        precision = len(pred_set & target_set) / (len(pred_set) + 0.01)
        recall = len(pred_set & target_set) / (len(target_set) + 0.01)
    else:
        pred_set = set(pred_pos.cpu().numpy())
        target_set = set(target_pos.reshape([-1]).cpu().numpy())
        precision = len(pred_set & target_set) / (len(pred_set) + 0.01)
        recall = len(pred_set & target_set) / (len(target_set) + 0.01)

    # Calculate TP, TN, FP, FN for MCC
    TP = len(pred_set & target_set)
    FP = len(pred_set - target_set)
    FN = len(target_set - pred_set)
    TN = target.shape[0] - TP - FP - FN  # Total number of samples minus TP, FP, FN

    # Convert TP, FP, TN, FN to tensors for proper computation
    TP = t.tensor(TP, dtype=t.float32)
    FP = t.tensor(FP, dtype=t.float32)
    FN = t.tensor(FN, dtype=t.float32)
    TN = t.tensor(TN, dtype=t.float32)

    # Calculate MCC
    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if denominator != 0:
        mcc = (TP * TN - FP * FN) / t.sqrt(denominator)
    else:
        mcc = t.tensor(0.0)  # Avoid division by zero

    # Calculate MCC
    # if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
    #     mcc = (TP * TN - FP * FN) / t.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # else:
    #     mcc = t.tensor(0.0)  # Avoid division by zero
    return precision, recall, acc, mcc


def evaluate_model(model, dataloader):
    model.eval()
    acc_list = []
    loss_list = []
    precision_list = []
    recall_list = []
    mcc_list = []  # To store MCC values
    with t.set_grad_enabled(False):
        for (rna_name, seq_matrix), dot_matrix in dataloader:
            print('Start eva:',rna_name)
            if seq_matrix.shape[2] > 1024:
                #print(rna_name,' len is',seq_matrix.shape[2],' > 1024,will skill')
                continue
            pred = model(seq_matrix)

            precision, recall, acc, mcc = precision_recall_accuracy(t.sigmoid(pred), dot_matrix)
            precision_list.append(precision)
            recall_list.append(recall)
            acc_list.append(acc)
            mcc_list.append(mcc)

            loss = LogisticLoss(pred, dot_matrix)
            loss_list.append(loss.item())

    model.train()
    return np.mean(acc_list), np.mean(loss_list), np.mean(precision_list), np.mean(recall_list), np.mean(mcc_list)

def plot_acc_loss(session_name, hist_acc, hist_loss, hist_preci, hist_recall, hist_mcc):
    train_acc = [d[0] for d in hist_acc]
    val_acc = [d[1] for d in hist_acc]
    train_loss = [d[0] for d in hist_loss]
    val_loss = [d[1] for d in hist_loss]
    train_preci = [d[0] for d in hist_preci]
    val_preci = [d[1] for d in hist_preci]
    train_recall = [d[0] for d in hist_recall]
    val_recall = [d[1] for d in hist_recall]
    train_mcc = [d[0] for d in hist_mcc]
    val_mcc = [d[1] for d in hist_mcc]

    plt.plot(range(1, 1 + len(train_acc)), train_acc, c='#e91e63', label="training acc")
    plt.plot(range(1, 1 + len(val_acc)), val_acc, c='#4caf50', label="validation acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title(session_name)
    plt.savefig(session_name + "_acc.png")
    plt.close()

    plt.plot(range(1, 1 + len(train_loss)), train_loss, c='#e91e63', label="training loss")
    plt.plot(range(1, 1 + len(val_loss)), val_loss, c='#4caf50', label="validation loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(session_name)
    plt.savefig(session_name + "_loss.png")
    plt.close()

    plt.plot(range(1, 1 + len(train_preci)), train_preci, c='#e91e63', label="training precision")
    plt.plot(range(1, 1 + len(val_preci)), val_preci, c='#4caf50', label="validation precision")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("precision")
    plt.title(session_name)
    plt.savefig(session_name + "_precision.png")
    plt.close()

    plt.plot(range(1, 1 + len(train_recall)), train_recall, c='#e91e63', label="training recall")
    plt.plot(range(1, 1 + len(val_recall)), val_recall, c='#4caf50', label="validation recall")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("recall")
    plt.title(session_name)
    plt.savefig(session_name + "recall.png")
    plt.close()

    plt.plot(range(1, 1 + len(train_mcc)), train_mcc, c='#e91e63', label="training recall")
    plt.plot(range(1, 1 + len(val_mcc)), val_mcc, c='#4caf50', label="validation recall")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("MCC")
    plt.title(session_name)
    plt.savefig(session_name + "MCC.png")
    plt.close()
