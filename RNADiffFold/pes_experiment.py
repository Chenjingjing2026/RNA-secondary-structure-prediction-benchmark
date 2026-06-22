# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common.utils import add_parent_path
from common.experiment import add_exp_args as add_exp_args_parent
from common.experiment import DiffusionExperiment
from common.data_utils import contact_map_masks
from common.loss_utils import bce_loss, evaluate_f1_precision_recall
from common.loss_utils import calculate_auc, calculate_mattews_correlation_coefficient,rna_evaluation,rna_evaluation_modified
add_parent_path(level=2)
from collections import defaultdict
TEST=True

def add_exp_args(parser):
    add_exp_args_parent(parser)

def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device,non_blocking=True)
        # return batch.to(device)
    elif isinstance(batch, dict):
        return {key: batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(item, device) for item in batch]
    else:
        return batch


def load_base_pairs_info():
    # 创建目标位置集合
    target_positions = defaultdict(set)
    # for pair_idx in range(len(bp1_list)):
    #     pos1 = bp1_list[pair_idx] - 1  # 转换为0-based索引
    #     pos2 = bp2_list[pair_idx] - 1  # 转换为0-based索引
    #     target_positions.add((pos1, pos2))

    base_pairs_dict1 = defaultdict(set)
    base_pairs_dict2 = defaultdict(set)
    base_pairs_file= '/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/pseudoknot/short/all_pk_pairs.txt'

    try:
        with open(base_pairs_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        filename = parts[0]
                        pos1 = int(parts[2])
                        pos2 = int(parts[4])
                        target_positions[filename].add((pos1, pos2))
                        # base_pairs_dict1[filename].add(pos1)
                        # base_pairs_dict2[filename].add(pos2)
    except Exception as e:
        print(f"Warning: Error loading base pairs file: {e}")
    return dict(target_positions)
    # return dict(base_pairs_dict1),dict(base_pairs_dict2)
target_positions = load_base_pairs_info()

# base_pair1,base_pair2 = load_base_pairs_info()
def evaluate_model_separate_bp(pred_a, true_a,name):
    pred_a = pred_a[0]
    print('pred_a',pred_a.shape)
    print('true_a', true_a.shape)


    target_tp, target_fp, target_tn, target_fn = 0, 0, 0, 0
    other_tp, other_fp, other_tn, other_fn = 0, 0, 0, 0

    name = name[0].replace('.bpseq','')
    print('name', name)


    # 遍历每个位置
    for j in range(pred_a.shape[0]):
        for i in range(pred_a.shape[1]):
            pred = pred_a[j, i]
            true_val = true_a[j, i]

            if (j+1, i+1) in target_positions[name]:
                # 目标位置的混淆矩阵更新
                if pred == 1 and true_val == 1:
                    target_tp += 1
                elif pred == 1 and true_val == 0:
                    target_fp += 1
                elif pred == 0 and true_val == 0:
                    target_tn += 1
                elif pred == 0 and true_val == 1:
                    target_fn += 1
                print(f'{name} position ({j+1},{i+1}): tp{target_tp}, fp{target_fp}, tn{target_tn}, fn{target_fn}')
            else:
                # 非目标位置的混淆矩阵更新
                if pred == 1 and true_val == 1:
                    other_tp += 1
                elif pred == 1 and true_val == 0:
                    other_fp += 1
                elif pred == 0 and true_val == 0:
                    other_tn += 1
                elif pred == 0 and true_val == 1:
                    other_fn += 1
    # 遍历每个位置
    # for j in range(pred_a.shape[0]):
    #     for i in range(pred_a.shape[1]):
    #         pred = pred_a[j, i]  # 应该是单个元素，而不是整行
    #         true_val = true_a[j, i]  # 应该是单个元素，而不是整行
    #         if name in base_pair1:
    #             # if j+1 in base_pair1[name] and i+1 in base_pair2[name]:
    #             if j + 1 in base_pair1[name] and (i + 1) == base_pair2[name][j]:
    #                     # 目标位置的混淆矩阵更新
    #                 if pred == 1 and true_val == 1:
    #                     target_tp += 1
    #                 elif pred == 1 and true_val == 0:
    #                     target_fp += 1
    #                 elif pred == 0 and true_val == 0:
    #                     target_tn += 1
    #                 elif pred == 0 and true_val == 1:
    #                     target_fn += 1
    #                 print(f'{name}  position ({j},{i}): tp{target_tp}, fp{target_fp}, tn{target_tn}, fn{target_fn}')
    #                 # else:
    #                 #     # 非目标碱基对位置的混淆矩阵更新
    #                 #     if pred == 1 and true_val == 1:
    #                 #         other_tp += 1
    #                 #     elif pred == 1 and true_val == 0:
    #                 #         other_fp += 1
    #                 #     elif pred == 0 and true_val == 0:
    #                 #         other_tn += 1
    #                 #     elif pred == 0 and true_val == 1:
    #                 #         other_fn += 1
    #             else:
    #                 # 非目标位置的混淆矩阵更新
    #                 if pred == 1 and true_val == 1:
    #                     other_tp += 1
    #                 elif pred == 1 and true_val == 0:
    #                     other_fp += 1
    #                 elif pred == 0 and true_val == 0:
    #                     other_tn += 1
    #                 elif pred == 0 and true_val == 1:
    #                     other_fn += 1

            # print('target_tp, target_fp, target_tn, target_fn', target_tp, target_fp, target_tn, target_fn)
            # print('other_tp, other_fp, other_tn, other_fn', other_tp, other_fp, other_tn, other_fn)
            # 计算特定碱基对的指标
            target_precision = target_tp / (target_tp + target_fp) if (target_tp + target_fp) > 0 else 0
            target_recall = target_tp / (target_tp + target_fn) if (target_tp + target_fn) > 0 else 0
            target_f1_score = (2 * target_tp) / (2 * target_tp + target_fp + target_fn) if (target_tp + target_fn) > 0 else 0

            # 计算其他碱基对的指标
            other_precision = other_tp / (other_tp + other_fp) if (other_tp + other_fp) > 0 else 0
            other_recall = other_tp / (other_tp + other_fn) if (other_tp + other_fn) > 0 else 0
            other_f1_score = (2 * other_tp) / (2 * other_tp + other_fp + other_fn) if (other_tp + other_fn) > 0 else 0

            # 计算MCC
            target_mcc_numerator = (target_tp * target_tn) - (target_fp * target_fn)
            target_mcc_denominator = ((target_tp + target_fp) * (target_tp + target_fn) *
                                      (target_tn + target_fp) * (target_tn + target_fn)) ** 0.5
            target_mcc = target_mcc_numerator / target_mcc_denominator if target_mcc_denominator > 0 else 0

            other_mcc_numerator = (other_tp * other_tn) - (other_fp * other_fn)
            other_mcc_denominator = ((other_tp + other_fp) * (other_tp + other_fn) *
                                     (other_tn + other_fp) * (other_tn + other_fn)) ** 0.5
            other_mcc = other_mcc_numerator / other_mcc_denominator if other_mcc_denominator > 0 else 0
        # else:
        #     print('no target_precision')
    return target_precision,target_recall,target_f1_score,target_mcc,other_precision,other_recall,other_f1_score,other_mcc



class pes_Experiment(DiffusionExperiment):
    def test_fn(self, epoch=0):
        self.model.eval()
        device = self.args.device
        with torch.no_grad():
            test_no_train = list()
            total_name_list = list()
            total_length_list = list()

            pes_precision_list = []
            pes_recall_list = []
            pes_f1_list = []
            pes_mcc_list = []

            other_precision_list = []
            other_recall_list = []
            other_f1_list = []
            other_mcc_list = []

            # 在循环开始前初始化列表
            pes_detailed_results = []
            other_detailed_results = []

            for _, batch in enumerate(
                    self.test_loader):
                batch=batch_to_device(batch, device)
                total_name_list += [item for item in batch['data_name']]
                total_length_list += [item.item() for item in batch['data_length']]
                matrix_rep = torch.zeros_like(batch['contact'])
                # contact = contact.to(device)
                # data_fcn_2 = data_fcn_2.to(device)
                # data_length = data_length.to(device)
                # data_seq_raw = data_seq_raw.to(device)
                # data_seq_encoding = data_seq_encoding.to(device)
                contact_masks = contact_map_masks(batch['data_length'], matrix_rep).to(device)  # data_length以内的为1
                contact_masks = contact_masks.unsqueeze(1)
                # calculate contact loss
                batch_size = batch['contact'].shape[0]
                pred_x0, _ = self.model.sample(batch_size, batch['data_fcn_2'], batch['tokens'], batch['set_max_len'], contact_masks,batch['data_seq_encode_pad'])
                pred_x0 = pred_x0.cpu().float()


                print('map_no_train', pred_x0.shape)
                target_precision, target_recall, target_f1_score, target_mcc, other_precision, other_recall, other_f1_score, other_mcc = evaluate_model_separate_bp(
                    pred_x0[0], batch['contact'].cpu().float()[0], batch['data_name'])

                pes_precision_list.append(target_precision)
                pes_recall_list.append(target_recall)
                # acc_list.append(acc.item())
                pes_f1_list.append(target_f1_score)
                pes_mcc_list.append(target_mcc)

                other_precision_list.append(other_precision)
                other_recall_list.append(other_recall)
                other_f1_list.append(other_f1_score)
                other_mcc_list.append(other_mcc)

                # 记录每个样本的指标 - 一行格式，空格分隔
                sample_name = batch['data_name'][0] if isinstance(batch['data_name'], list) else batch['data_name']
                pes_detailed_results.append(
                    f"{sample_name} {target_precision:.4f} {target_recall:.4f} {target_f1_score:.4f} {target_mcc:.4f}")
                other_detailed_results.append(
                    f"{sample_name} {other_precision:.4f} {other_recall:.4f} {other_f1_score:.4f} {other_mcc:.4f}")

            target_metrics = (np.mean(pes_f1_list), np.mean(pes_precision_list), np.mean(pes_recall_list),
                              np.mean(pes_mcc_list))
            other_metrics = (np.mean(other_f1_list), np.mean(other_precision_list), np.mean(other_recall_list),
                             np.mean(other_mcc_list))

            target_F1, target_preci, target_recall, target_mcc = target_metrics
            other_F1, other_preci, other_recall, other_mcc = other_metrics

            # 写入Target Base Pairs指标到文件
            output_file_target = '/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/pes1/target_base_pairs_metrics.txt'
            os.makedirs(os.path.dirname(output_file_target), exist_ok=True)
            with open(output_file_target, 'w') as f:
                for result in pes_detailed_results:
                    f.write(result + '\n')

            # 写入Other Base Pairs指标到文件
            output_file_other = '/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/pes1/other_base_pairs_metrics.txt'
            os.makedirs(os.path.dirname(output_file_other), exist_ok=True)
            with open(output_file_other, 'w') as f:
                for result in other_detailed_results:
                    f.write(result + '\n')

            print("\n=== Target Base Pairs Results ===")
            print("test_preci=%.5f, test_recall=%.5f, test_F1=%.5f, test_MCC=%.5f" % (
                target_preci, target_recall, target_F1, target_mcc))

            print("\n=== Other Base Pairs Results ===")
            print("test_preci=%.5f, test_recall=%.5f, test_F1=%.5f, test_MCC=%.5f" % (
                other_preci, other_recall, other_F1, other_mcc))
