# import time
# import torch
# import numpy as np
# from tqdm import tqdm
# from utils import cuda
# from API import evaluate_result
# from model import RFold_Model
#
#
# # predefine a base_matrix
# _max_length = 1005
# base_matrix = torch.ones(_max_length, _max_length)
# for i in range(_max_length):
#     st, en = max(i-3, 0), min(i+3, _max_length-1)
#     for j in range(st, en + 1):
#         base_matrix[i, j] = 0.
#
# def constraint_matrix(x):
#     base_a, base_u, base_c, base_g = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
#     batch = base_a.shape[0]
#     length = base_a.shape[1]
#     au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
#     au_ua = au + torch.transpose(au, -1, -2)
#     cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
#     cg_gc = cg + torch.transpose(cg, -1, -2)
#     ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
#     ug_gu = ug + torch.transpose(ug, -1, -2)
#     return (au_ua + cg_gc + ug_gu) * base_matrix[:length, :length].to(x.device)
#
# def row_col_softmax(y):
#     row_softmax = torch.softmax(y, dim=-1)
#     col_softmax = torch.softmax(y, dim=-2)
#     return 0.5 * (row_softmax + col_softmax)
#
# def row_col_argmax(y):
#     y_pred = row_col_softmax(y)
#     y_hat = y_pred + torch.randn_like(y) * 1e-12
#     col_max = torch.argmax(y_hat, 1)
#     col_one = torch.zeros_like(y_hat).scatter(1, col_max.unsqueeze(1), 1.0)
#     row_max = torch.argmax(y_hat, 2)
#     row_one = torch.zeros_like(y_hat).scatter(2, row_max.unsqueeze(2), 1.0)
#     int_one = row_one * col_one
#     return int_one
#
#
# class RFold(object):
#     def __init__(self, args, device):
#         self.args = args
#         self.device = device
#         self.config = args.__dict__
#
#         self.model = self._build_model()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
#         self.criterion = torch.nn.MSELoss()
#
#     def _build_model(self, **kwargs):
#         return RFold_Model(self.args).to(self.device)
#
#     def test_one_epoch(self, test_loader, **kwargs):
#         # note that the model is under the training mode for bn/dropout
#         self.model.train()
#         eval_results, run_time = [], []
#         test_pbar = tqdm(test_loader)
#         for batch in test_pbar:
#             contacts, seq_lens, seq_ori = batch
#             contacts, seq_ori = cuda(
#                 (contacts.float(), seq_ori.float()), device=self.device)
#
#             # predict
#             seqs = torch.argmax(seq_ori, axis=-1)
#             s_time = time.time()
#             with torch.no_grad():
#                 pred_contacts = self.model(seqs)
#
#             pred_contacts = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)
#
#             # interval time
#             interval_t = time.time() - s_time
#             run_time.append(interval_t)
#
#             eval_result = list(map(lambda i: evaluate_result(pred_contacts.cpu()[i],
#                                                                      contacts.cpu()[i]), range(contacts.shape[0])))
#             eval_results += eval_result
#
#         p, r, f1, mcc = zip(*eval_results)
#         return np.average(f1), np.average(p), np.average(r), np.average(mcc), np.average(run_time)
#
#     def process_batch(self, batch):
#         """
#         处理数据批次，返回模型输入和目标
#         :param batch: 数据批次（通常是一个列表，包含多个样本）
#         :return: inputs (模型输入), targets (目标值)
#         """
#         # 假设 batch 是一个列表，每个样本是 [contact, seq_length, padded_seq, name, pairs]
#         print(batch[0].shape)
#         contacts = torch.stack([torch.tensor(x[0]) for x in batch]).to(self.device)
#         seqs = torch.stack([torch.tensor(x[2]) for x in batch]).to(self.device)
#         lengths = [x[1] for x in batch]  # 各样本的实际长度
#
#         # 创建 mask（可选，用于处理变长序列）
#         max_len = contacts.size(1)
#         mask = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]
#         mask = mask.to(self.device)
#
#         return (seqs, mask), contacts  # 返回模型输入和目标
#
#     def train_one_epoch(self, train_loader, **kwargs):
#         # 设置模型为训练模式（启用Dropout和BatchNorm）
#         self.model.train()
#
#         # 初始化统计指标
#         total_loss = 0.0
#         eval_results = []
#         run_time = []
#
#         # 进度条包装训练数据加载器
#         train_pbar = tqdm(train_loader)
#
#         for batch in train_pbar:
#             # ----------------------------
#             # 1. 数据准备
#             # ----------------------------
#             contacts, seq_lens, seq_ori = batch
#
#             # 将数据转移到GPU并转换类型
#             contacts = cuda(contacts.float(), device=self.device)
#             seq_ori = cuda(seq_ori.float(), device=self.device)
#
#             # ----------------------------
#             # 2. 梯度清零
#             # ----------------------------
#             self.optimizer.zero_grad()
#
#             # ----------------------------
#             # 3. 前向传播
#             # ----------------------------
#             # 将one-hot序列转换为索引（假设模型需要离散输入）
#             seqs = torch.argmax(seq_ori, dim=-1)
#
#             # 记录前向传播时间（可选）
#             s_time = time.time()
#             pred_contacts = self.model(seqs)  # 模型预测
#             fwd_time = time.time() - s_time
#             run_time.append(fwd_time)
#
#             # ----------------------------
#             # 4. 损失计算
#             # ----------------------------
#             # 使用二值交叉熵损失（假设pred_contacts是logits）
#             loss = self.criterion(pred_contacts, contacts)
#
#             # ----------------------------
#             # 5. 反向传播与优化
#             # ----------------------------
#             loss.backward()
#             self.optimizer.step()
#
#             # ----------------------------
#             # 6. 统计记录
#             # ----------------------------
#             total_loss += loss.item()
#
#             # 计算评估指标（非必需，会增加计算开销）
#             with torch.no_grad():
#                 # 应用与测试时相同的后处理
#                 processed_pred = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)
#
#                 # 批量计算评估结果
#                 batch_eval = [
#                     evaluate_result(processed_pred[i].cpu(), contacts[i].cpu())
#                     for i in range(contacts.shape[0])
#                 ]
#                 eval_results.extend(batch_eval)
#
#             # 更新进度条描述
#             train_pbar.set_description(
#                 f"Loss: {loss.item():.4f} | "
#                 f"F1: {np.mean([f for _, _, f, _ in batch_eval]):.4f}"
#             )
#
#         # ----------------------------
#         # 7. 汇总epoch结果
#         # ----------------------------
#         avg_loss = total_loss / len(train_loader)
#         avg_time = np.mean(run_time)
#
#         # 解包评估结果
#         if eval_results:
#             p, r, f1, mcc = zip(*eval_results)
#             return avg_loss, np.mean(f1), np.mean(p), np.mean(r), np.mean(mcc), avg_time
#         else:
#             return avg_loss, 0.0, 0.0, 0.0, avg_time

import time
import torch
import numpy as np
from tqdm import tqdm
from utils import cuda
from API import evaluate_result
from model import RFold_Model
import os


# predefine a base_matrix
_max_length = 1005
base_matrix = torch.ones(_max_length, _max_length)
for i in range(_max_length):
    st, en = max(i-3, 0), min(i+3, _max_length-1)
    for j in range(st, en + 1):
        base_matrix[i, j] = 0.

def constraint_matrix(x):
    base_a, base_u, base_c, base_g = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return_value = None

    # try:
    #     return_value = (au_ua + cg_gc + ug_gu) * base_matrix[:length, :length].to(x.device)
    # except Exception as e:
    #     # 打印维度信息
    #     print(f"length = {length}")
    #     print(f"au_ua.shape = {au_ua.shape}")
    #     print(f"cg_gc.shape = {cg_gc.shape}")
    #     print(f"ug_gu.shape = {ug_gu.shape}")
    #     print(f"base_matrix[:length, :length].shape = {base_matrix[:length, :length].shape}")
    if length > _max_length:
        print(f"警告: 序列长度 {length} 超过 base_matrix 最大维度 {_max_length}，将自动截断")
        return_value = (au_ua[:,:_max_length, :_max_length] + cg_gc[:,:_max_length, :_max_length] + ug_gu[:,:_max_length, :_max_length]) * base_matrix[:_max_length, :_max_length].to(x.device)
    else:
        return_value = (au_ua + cg_gc + ug_gu) * base_matrix[:length, :length].to(x.device)

    return return_value

def row_col_softmax(y):
    row_softmax = torch.softmax(y, dim=-1)
    col_softmax = torch.softmax(y, dim=-2)
    return 0.5 * (row_softmax + col_softmax)

def row_col_argmax(y):
    y_pred = row_col_softmax(y)
    y_hat = y_pred + torch.randn_like(y) * 1e-12
    col_max = torch.argmax(y_hat, 1)
    col_one = torch.zeros_like(y_hat).scatter(1, col_max.unsqueeze(1), 1.0)
    row_max = torch.argmax(y_hat, 2)
    row_one = torch.zeros_like(y_hat).scatter(2, row_max.unsqueeze(2), 1.0)
    int_one = row_one * col_one
    return int_one

from collections import defaultdict
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

    except Exception as e:
        print(f"Warning: Error loading base pairs file: {e}")
    return dict(target_positions)

target_positions = load_base_pairs_info()

def evaluate_model_separate_bp(pred_a, true_a,name):
    target_correct = 0
    target_total = 0
    target_tp, target_fp, target_tn, target_fn = 0, 0, 0, 0

    # 初始化其他碱基对的指标
    other_total_loss = 0.0
    other_correct = 0
    other_total = 0
    other_tp, other_fp, other_tn, other_fn = 0, 0, 0, 0

    name = name[0].replace('.bpseq','')
    # pred_a = pred_a.cpu().reshape(-1)
    # true_a = true_a.cpu().reshape(-1)

    # 遍历每个位置
    for j in range(pred_a.shape[0]):
        for i in range(pred_a.shape[1]):
            pred = pred_a[j, i]
            true_val = true_a[j, i]

            if (j + 1, i + 1) in target_positions[name]:
                # 目标位置的混淆矩阵更新
                if pred == 1 and true_val == 1:
                    target_tp += 1
                elif pred == 1 and true_val == 0:
                    target_fp += 1
                elif pred == 0 and true_val == 0:
                    target_tn += 1
                elif pred == 0 and true_val == 1:
                    target_fn += 1
                print(f'{name} position ({j + 1},{i + 1}): tp{target_tp}, fp{target_fp}, tn{target_tn}, fn{target_fn}')
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


            # 计算特定碱基对的指标
            # target_accuracy = target_correct / target_total if target_total > 0 else 0
            # target_avg_loss = target_total_loss / target_total if target_total > 0 else 0
            target_precision = target_tp / (target_tp + target_fp) if (target_tp + target_fp) > 0 else 0
            target_recall = target_tp / (target_tp + target_fn) if (target_tp + target_fn) > 0 else 0
            target_f1_score = (2 * target_tp) / (2 * target_tp + target_fp + target_fn) if (target_tp + target_fn) > 0 else 0

            # 计算其他碱基对的指标
            # other_accuracy = other_correct / other_total if other_total > 0 else 0
            # other_avg_loss = other_total_loss / other_total if other_total > 0 else 0
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



class RFold(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.config = args.__dict__

        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = torch.nn.MSELoss()

    def _build_model(self, **kwargs):
        return RFold_Model(self.args).to(self.device)

    def test_one_epoch_record(self, test_loader, **kwargs):
        # note that the model is under the training mode for bn/dropout
        self.model.train()
        eval_results, run_time = [], []
        show_progress = False  # 用变量控制是否显示进度条
        test_pbar = tqdm(test_loader,disable = not show_progress)

        # 用于存储每个样本的详细结果
        detailed_results = []

        for batch in test_pbar:
            contacts, seq_lens, seq_ori,name = batch
            if seq_ori.shape[1] > 1005:
                continue
            contacts, seq_ori = cuda(
                (contacts.float(), seq_ori.float()), device=self.device)

            # predict
            seqs = torch.argmax(seq_ori, axis=-1)
            s_time = time.time()
            with torch.no_grad():
                pred_contacts = self.model(seqs)
            pred_contacts = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)

            # interval time
            interval_t = time.time() - s_time
            run_time.append(interval_t)

            eval_result = list(map(lambda i: evaluate_result(pred_contacts.cpu()[i],
                                                                     contacts.cpu()[i]), range(contacts.shape[0])))
            eval_results += eval_result

            # 记录每个样本的详细指标 - 表格格式
            for i, (p_val, r_val, f1_val, mcc_val) in enumerate(eval_result):
                sample_name = name[i] if isinstance(name, list) else name
                detailed_results.append(f"{sample_name} {p_val:.4f} {r_val:.4f} {f1_val:.4f} {mcc_val:.4f}")

        p, r, f1, mcc = zip(*eval_results)

        # 写入详细结果到txt文件
        output_file = '/home/chenjingjing/Models/RFold/RFold-master/results/length/medium.txt'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for result in detailed_results:
                f.write(result + '\n')

        return np.average(f1), np.average(p), np.average(r), np.average(mcc), np.average(run_time)

    def test_one_epoch(self, test_loader, **kwargs):
        # note that the model is under the training mode for bn/dropout
        self.model.train()
        eval_results, run_time = [], []
        show_progress = False  # 用变量控制是否显示进度条
        test_pbar = tqdm(test_loader, disable=not show_progress)

        for batch in test_pbar:
            contacts, seq_lens, seq_ori, name = batch
            if seq_ori.shape[1] > 1005:
                continue
            contacts, seq_ori = cuda(
                (contacts.float(), seq_ori.float()), device=self.device)

            # predict
            seqs = torch.argmax(seq_ori, axis=-1)
            s_time = time.time()
            with torch.no_grad():
                pred_contacts = self.model(seqs)
            pred_contacts = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)

            # interval time
            interval_t = time.time() - s_time
            run_time.append(interval_t)

            eval_result = list(map(lambda i: evaluate_result(pred_contacts.cpu()[i],
                                                             contacts.cpu()[i]), range(contacts.shape[0])))
            eval_results += eval_result

        p, r, f1, mcc = zip(*eval_results)

        return np.average(f1), np.average(p), np.average(r), np.average(mcc), np.average(run_time)

    def pes_test(self, test_loader, **kwargs):
        # note that the model is under the training mode for bn/dropout
        self.model.train()
        eval_results, run_time = [], []

        pes_precision_list = []
        pes_recall_list = []
        pes_f1_list = []
        pes_mcc_list = []

        other_precision_list = []
        other_recall_list = []
        other_f1_list = []
        other_mcc_list = []

        # 用于存储每个样本的详细结果
        pes_detailed_results = []
        other_detailed_results = []

        show_progress = False  # 用变量控制是否显示进度条
        test_pbar = tqdm(test_loader,disable = not show_progress)
        for batch in test_pbar:
            contacts, seq_lens, seq_ori,name = batch
            if seq_ori.shape[1] > 1005:
                continue
            contacts, seq_ori = cuda(
                (contacts.float(), seq_ori.float()), device=self.device)

            # predict
            seqs = torch.argmax(seq_ori, axis=-1)
            s_time = time.time()
            with torch.no_grad():
                pred_contacts = self.model(seqs)
            pred_contacts = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)

            # interval time
            interval_t = time.time() - s_time
            run_time.append(interval_t)


            target_precision, target_recall, target_f1_score, target_mcc, other_precision, other_recall, other_f1_score, other_mcc = evaluate_model_separate_bp(
                pred_contacts.cpu()[0],contacts.cpu()[0], name)

            # 记录每个样本的指标 - 表格格式
            sample_name = name[0] if isinstance(name, list) else name
            pes_detailed_results.append(
                f"{sample_name} {target_precision:.4f} {target_recall:.4f} {target_f1_score:.4f} {target_mcc:.4f}")
            other_detailed_results.append(
                f"{sample_name} {other_precision:.4f} {other_recall:.4f} {other_f1_score:.4f} {other_mcc:.4f}")

            pes_precision_list.append(target_precision)
            pes_recall_list.append(target_recall)
            # acc_list.append(acc.item())
            pes_f1_list.append(target_f1_score)
            pes_mcc_list.append(target_mcc)

            other_precision_list.append(other_precision)
            other_recall_list.append(other_recall)
            other_f1_list.append(other_f1_score)
            other_mcc_list.append(other_mcc)

        target_metrics = (np.mean(pes_f1_list), np.mean(pes_precision_list), np.mean(pes_recall_list),
                          np.mean(pes_mcc_list))
        other_metrics = (np.mean(other_f1_list), np.mean(other_precision_list), np.mean(other_recall_list),
                         np.mean(other_mcc_list))

        # 写入PES metrics - 只写每个样本的数据，不写平均值
        output_file_pes = '/home/chenjingjing/Models/RFold/RFold-master/results/pes/pes_test_metrics.txt'
        os.makedirs(os.path.dirname(output_file_pes), exist_ok=True)
        with open(output_file_pes, 'w') as f:
            for result in pes_detailed_results:
                f.write(result + '\n')

        # 写入Other metrics - 只写每个样本的数据，不写平均值
        output_file_other = '/home/chenjingjing/Models/RFold/RFold-master/results/pes/other_test_metrics.txt'
        os.makedirs(os.path.dirname(output_file_other), exist_ok=True)
        with open(output_file_other, 'w') as f:
            for result in other_detailed_results:
                f.write(result + '\n')

        return target_metrics,other_metrics


    def process_batch(self, batch):
        """
        处理数据批次，返回模型输入和目标
        :param batch: 数据批次（通常是一个列表，包含多个样本）
        :return: inputs (模型输入), targets (目标值)
        """
        # 假设 batch 是一个列表，每个样本是 [contact, seq_length, padded_seq, name, pairs]
        print(batch[0].shape)
        contacts = torch.stack([torch.tensor(x[0]) for x in batch]).to(self.device)
        seqs = torch.stack([torch.tensor(x[2]) for x in batch]).to(self.device)
        lengths = [x[1] for x in batch]  # 各样本的实际长度
        names = [x[3] for x in batch]  # 各样本的实际长度

        # 创建 mask（可选，用于处理变长序列）
        max_len = contacts.size(1)
        mask = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]
        mask = mask.to(self.device)

        return (seqs, mask), contacts  # 返回模型输入和目标

    def train_one_epoch(self, train_loader, **kwargs):
        # 设置模型为训练模式（启用Dropout和BatchNorm）
        self.model.train()

        # 初始化统计指标
        total_loss = 0.0
        eval_results = []
        run_time = []
        show_progress = False  # 用变量控制是否显示进度条
        # 进度条包装训练数据加载器
        train_pbar = tqdm(train_loader,disable = not show_progress)

        for batch in train_pbar:
            # ----------------------------
            # 1. 数据准备
            # ----------------------------
            contacts, seq_lens, seq_ori ,name = batch
            if seq_ori.shape[1] > 1005:
                continue

            # 将数据转移到GPU并转换类型
            contacts = cuda(contacts.float(), device=self.device)
            seq_ori = cuda(seq_ori.float(), device=self.device)

            # ----------------------------
            # 2. 梯度清零
            # ----------------------------
            self.optimizer.zero_grad()

            # ----------------------------
            # 3. 前向传播
            # ----------------------------
            # 将one-hot序列转换为索引（假设模型需要离散输入）
            seqs = torch.argmax(seq_ori, dim=-1)

            # 记录前向传播时间（可选）
            s_time = time.time()
            pred_contacts = self.model(seqs)  # 模型预测
            fwd_time = time.time() - s_time
            run_time.append(fwd_time)

            # ----------------------------
            # 4. 损失计算
            # ----------------------------
            # 使用二值交叉熵损失（假设pred_contacts是logits）
            loss = self.criterion(pred_contacts, contacts)

            # ----------------------------
            # 5. 反向传播与优化
            # ----------------------------
            loss.backward()
            self.optimizer.step()

            # ----------------------------
            # 6. 统计记录
            # ----------------------------
            total_loss += loss.item()

            # 计算评估指标（非必需，会增加计算开销）
            with torch.no_grad():
                # 应用与测试时相同的后处理
                try:
                    processed_pred = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)
                except:
                    print('seq_ori', seq_ori.shape)
                    print('seq_len', seq_lens)
                    print('name',name)
                # 批量计算评估结果
                batch_eval = [
                    evaluate_result(processed_pred[i].cpu(), contacts[i].cpu())
                    for i in range(contacts.shape[0])
                ]
                eval_results.extend(batch_eval)

            # 更新进度条描述
            train_pbar.set_description(
                f"Loss: {loss.item():.4f} | "
                f"F1: {np.mean([f for _, _, f, _ in batch_eval]):.4f}"
            )

        # ----------------------------
        # 7. 汇总epoch结果
        # ----------------------------
        avg_loss = total_loss / len(train_loader)
        avg_time = np.mean(run_time)

        # 解包评估结果
        if eval_results:
            p, r, f1, mcc = zip(*eval_results)
            return avg_loss, np.mean(f1), np.mean(p), np.mean(r), np.mean(mcc), avg_time
        else:
            return avg_loss, 0.0, 0.0, 0.0, avg_time