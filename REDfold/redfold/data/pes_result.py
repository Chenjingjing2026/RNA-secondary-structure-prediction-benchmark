from collections import defaultdict
import re
import os
import numpy as np
import torch


def load_base_pairs_info():
    base_pairs_dict = defaultdict(set)
    base_pairs_file = '/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/pseudoknot/short/all_pk_pairs.txt'

    try:
        with open(base_pairs_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        filename = parts[0]
                        pos1 = int(parts[2])
                        pos2 = int(parts[4])
                        base_pairs_dict[filename].add(pos1)
                        base_pairs_dict[filename].add(pos2)
    except Exception as e:
        print(f"Warning: Error loading base pairs file: {e}")
    return dict(base_pairs_dict)


base_pair = load_base_pairs_info()


# Define a function to parse each file and extract the necessary information
def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regex to extract the relevant data: ID and Pairing line
    id_pattern = r'>([^ \n]+)'  # Extract everything after ">"
    pairing_pattern = r'Pairing:\s([0-9\s]+)'  # Extract the numbers after "Pairing:"

    # Find all matches using the regex
    ids = re.findall(id_pattern, content)
    pairings = re.findall(pairing_pattern, content)

    # Create a dictionary to store the results
    pred = {}

    # Iterate over all matches and extract the file ID and corresponding pair tensor
    for file_id, pairing in zip(ids, pairings):
        pair_tensor = np.array([int(i) for i in pairing.split()])
        pred[file_id] = torch.tensor(pair_tensor)

    return pred


def parse_bpseq_files(bpseq_folder):
    # 获取 bpseq 文件夹中的所有文件
    bpseq_files = [f for f in os.listdir(bpseq_folder) if f.endswith('.bpseq')]

    # 创建字典来存储数据
    ref = {}

    # 读取每个 bpseq 文件
    for filename in bpseq_files:
        # 构造文件路径
        bpseq_file_path = os.path.join(bpseq_folder, filename)

        # 打开并读取 bpseq 文件
        with open(bpseq_file_path, 'r') as bpseq_file:
            lines = bpseq_file.readlines()

        # 提取文件名作为 ID，去掉扩展名
        file_id = filename.replace('.bpseq', '')

        # 提取第三列的配对信息
        pairs = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 2:
                try:
                    pairs.append(int(parts[2]))
                except ValueError:
                    pairs.append(0)  # 如果转换失败，使用0

        # 转换为 tensor 并存入字典
        ref[file_id] = torch.tensor(pairs)

    return ref


# Ground truth
bpseq_folder = '/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/pseudoknot/short/test'
ref = parse_bpseq_files(bpseq_folder)

# 预测结果
file_path = '/home/chenjingjing/Models/REDfold1/redfold/data/pes/test.txt'
pred = parse_file(file_path)

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

for file_id, pair_tensor in pred.items():
    # 检查文件是否在参考数据中
    if file_id not in ref:
        print(f"Warning: {file_id} not found in reference data, skipping")
        continue

    # 初始化计数器
    target_tp, target_fp, target_tn, target_fn = 0, 0, 0, 0
    other_tp, other_fp, other_tn, other_fn = 0, 0, 0, 0

    ref_tensor = ref[file_id]

    for i in range(len(pair_tensor)):
        pred_val = pair_tensor[i].item()
        true_val = ref_tensor[i].item()

        # 判断是否是目标位置
        is_target = (file_id in base_pair and i+1 in base_pair[file_id])

        # 根据四种情况更新计数器
        if pred_val != 0 and true_val != 0:
            # 都有配对
            if pred_val == true_val:
                # 配对正确
                if is_target:
                    target_tp += 1
                else:
                    other_tp += 1
            else:
                # 配对错误
                if is_target:
                    target_fn += 1
                else:
                    other_fn += 1
        elif pred_val != 0 and true_val == 0:
            # 错误预测了配对
            if is_target:
                target_fp += 1
                print('fp,  pred_val, true_val', i, file_id, pred_val, true_val, target_fp)
            else:
                other_fp += 1
        elif pred_val == 0 and true_val != 0:
            # 漏掉了真实的配对
            if is_target:
                target_fn += 1
            else:
                other_fn += 1
        else:  # pred_val == 0 and true_val == 0
            # 正确预测无配对
            if is_target:
                target_tn += 1
                print('tn,  pred_val, true_val', i, file_id, pred_val, true_val, target_tn)
            else:
                other_tn += 1

    print('target_tp, target_fp, target_tn, target_fn', file_id, target_tp, target_fp, target_tn, target_fn)


    # 计算指标（应该在循环外部）
    target_precision = target_tp / (target_tp + target_fp) if (target_tp + target_fp) > 0 else 0
    target_recall = target_tp / (target_tp + target_fn) if (target_tp + target_fn) > 0 else 0
    target_f1_score = (2 * target_tp) / (2 * target_tp + target_fp + target_fn) if (
                                                                                               2 * target_tp + target_fp + target_fn) > 0 else 0

    other_precision = other_tp / (other_tp + other_fp) if (other_tp + other_fp) > 0 else 0
    other_recall = other_tp / (other_tp + other_fn) if (other_tp + other_fn) > 0 else 0
    other_f1_score = (2 * other_tp) / (2 * other_tp + other_fp + other_fn) if (
                                                                                          2 * other_tp + other_fp + other_fn) > 0 else 0

    # 计算MCC
    target_mcc_numerator = (target_tp * target_tn) - (target_fp * target_fn)
    target_mcc_denominator = ((target_tp + target_fp) * (target_tp + target_fn) * (target_tn + target_fp) * (
                target_tn + target_fn)) ** 0.5
    target_mcc = target_mcc_numerator / target_mcc_denominator if target_mcc_denominator > 0 else 0

    other_mcc_numerator = (other_tp * other_tn) - (other_fp * other_fn)
    other_mcc_denominator = ((other_tp + other_fp) * (other_tp + other_fn) * (other_tn + other_fp) * (
                other_tn + other_fn)) ** 0.5
    other_mcc = other_mcc_numerator / other_mcc_denominator if other_mcc_denominator > 0 else 0

    # 添加到列表
    pes_precision_list.append(target_precision)
    pes_recall_list.append(target_recall)
    pes_f1_list.append(target_f1_score)
    pes_mcc_list.append(target_mcc)

    other_precision_list.append(other_precision)
    other_recall_list.append(other_recall)
    other_f1_list.append(other_f1_score)
    other_mcc_list.append(other_mcc)

    # 记录每个样本的指标 - 表格格式，空格分隔
    pes_detailed_results.append(
        f"{file_id} {target_precision:.4f} {target_recall:.4f} {target_f1_score:.4f} {target_mcc:.4f}")
    other_detailed_results.append(
        f"{file_id} {other_precision:.4f} {other_recall:.4f} {other_f1_score:.4f} {other_mcc:.4f}")

# 计算平均指标
target_metrics = (np.mean(pes_f1_list), np.mean(pes_precision_list), np.mean(pes_recall_list), np.mean(pes_mcc_list))
other_metrics = (np.mean(other_f1_list), np.mean(other_precision_list), np.mean(other_recall_list),
                 np.mean(other_mcc_list))

target_F1, target_preci, target_recall, target_mcc = target_metrics
other_F1, other_preci, other_recall, other_mcc = other_metrics

# 写入Target Base Pairs指标到文件
output_file_target = '/home/chenjingjing/Models/REDfold1/redfold/data/pes/target_base_pairs_metrics.txt'
os.makedirs(os.path.dirname(output_file_target), exist_ok=True)
with open(output_file_target, 'w') as f:
    for result in pes_detailed_results:
        f.write(result + '\n')

# 写入Other Base Pairs指标到文件
output_file_other = '/home/chenjingjing/Models/REDfold1/redfold/data/pes/other_base_pairs_metrics.txt'
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







# from collections import defaultdict
#
# def load_base_pairs_info():
#
#     base_pairs_dict = defaultdict(set)
#     base_pairs_file= '/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/pseudoknot/short/all_pk_pairs.txt'
#
#     try:
#         with open(base_pairs_file, 'r') as f:
#             for line in f:
#                 if line.strip():
#                     parts = line.strip().split()
#                     if len(parts) >= 5:
#                         filename = parts[0]
#                         pos1 = int(parts[2])
#                         pos2 = int(parts[4])
#                         base_pairs_dict[filename].add(pos1)
#                         base_pairs_dict[filename].add(pos2)
#     except Exception as e:
#         print(f"Warning: Error loading base pairs file: {e}")
#     return dict(base_pairs_dict)
#
#
# base_pair= load_base_pairs_info()
#
# # Define a function to parse each file and extract the necessary information
# def parse_file(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()
#
#     # Use regex to extract the relevant data: ID and Pairing line
#     id_pattern = r'>([^ \n]+)'  # Extract everything after ">"
#     pairing_pattern = r'Pairing:\s([0-9\s]+)'  # Extract the numbers after "Pairing:"
#
#     # Find all matches using the regex
#     ids = re.findall(id_pattern, content)
#     pairings = re.findall(pairing_pattern, content)
#
#     # Create a dictionary to store the results
#     pred = {}
#
#     # Iterate over all matches and extract the file ID and corresponding pair tensor
#     for file_id, pairing in zip(ids, pairings):
#         pair_tensor = np.array([int(i) for i in pairing.split()])
#         pred[file_id] = torch.tensor(pair_tensor)
#
#     return pred
#
# def parse_bpseq_files(bpseq_folder):
#     # 获取 bpseq 文件夹中的所有文件
#     bpseq_files = [f for f in os.listdir(bpseq_folder) if f.endswith('.bpseq')]
#
#     # 创建字典来存储数据
#     ref = {}
#
#     # 读取每个 bpseq 文件
#     for filename in bpseq_files:
#         # 构造文件路径
#         bpseq_file_path = os.path.join(bpseq_folder, filename)
#
#         # 打开并读取 bpseq 文件
#         with open(bpseq_file_path, 'r') as bpseq_file:
#             lines = bpseq_file.readlines()
#
#         # 提取文件名作为 ID，去掉扩展名
#         file_id = filename.replace('.bpseq', '')
#
#         # 提取第三列的配对信息
#         pairs = [int(line.split()[2]) for line in lines if len(line.split()) > 2]
#
#         # 转换为 tensor 并存入字典
#         ref[file_id] = torch.tensor(pairs)
#
#     return ref
#
# # Ground truth
# bpseq_folder = '/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/pseudoknot/short/test'  # 替换为你的 bpseq 文件夹路径
# ref = parse_bpseq_files(bpseq_folder)
#
#
# # 预测结果
# file_path = '/home/chenjingjing/Models/REDfold1/redfold/data/pes/test.txt'  # replace with your file path
# pred = parse_file(file_path)
#
# pes_precision_list = []
# pes_recall_list = []
# pes_f1_list = []
# pes_mcc_list = []
#
# other_precision_list = []
# other_recall_list = []
# other_f1_list = []
# other_mcc_list = []
#
#
# for file_id, pair_tensor in pred.items():
#     target_tp, target_fp, target_tn, target_fn = 0, 0, 0, 0
#     other_tp, other_fp, other_tn, other_fn = 0, 0, 0, 0
#     # print(f"ID: {file_id}, Pair Tensor: {pair_tensor}")
#     if file_id in base_pair:
#         for i in len(pair_tensor):
#             if i in base_pair[file_id]:
#                 # 目标位置的混淆矩阵更新
#                 if pair_tensor[i] == 1 and ref[file_id][i] == 1:
#                     target_tp += 1
#                 elif pair_tensor[i] == 1 and ref[file_id][i] == 0:
#                     target_fp += 1
#                 elif pair_tensor[i] == 0 and ref[file_id][i] == 0:
#                     target_tn += 1
#                 elif pair_tensor[i] == 0 and ref[file_id][i] == 1:
#                     target_fn += 1
#             else:
#                 # 目标位置的混淆矩阵更新
#                 if pair_tensor[i] == 1 and ref[file_id][i] == 1:
#                     other_tp += 1
#                 elif pair_tensor[i] == 1 and ref[file_id][i] == 0:
#                     other_fp += 1
#                 elif pair_tensor[i] == 0 and ref[file_id][i] == 0:
#                     other_tn += 1
#                 elif pair_tensor[i] == 0 and ref[file_id][i] == 1:
#                     other_fn += 1
# # 计算特定碱基对的指标
#         target_precision = target_tp / (target_tp + target_fp) if (target_tp + target_fp) > 0 else 0
#         target_recall = target_tp / (target_tp + target_fn) if (target_tp + target_fn) > 0 else 0
#         target_f1_score = (2 * target_tp) / (2 * target_tp + target_fp + target_fn) if (target_tp + target_fn) > 0 else 0
#
#         # 计算其他碱基对的指标
#         other_precision = other_tp / (other_tp + other_fp) if (other_tp + other_fp) > 0 else 0
#         other_recall = other_tp / (other_tp + other_fn) if (other_tp + other_fn) > 0 else 0
#         other_f1_score = (2 * other_tp) / (2 * other_tp + other_fp + other_fn) if (other_tp + other_fn) > 0 else 0
#
#             # 计算MCC
#         target_mcc_numerator = (target_tp * target_tn) - (target_fp * target_fn)
#         target_mcc_denominator = ((target_tp + target_fp) * (target_tp + target_fn) *
#                                       (target_tn + target_fp) * (target_tn + target_fn)) ** 0.5
#         target_mcc = target_mcc_numerator / target_mcc_denominator if target_mcc_denominator > 0 else 0
#
#         other_mcc_numerator = (other_tp * other_tn) - (other_fp * other_fn)
#         other_mcc_denominator = ((other_tp + other_fp) * (other_tp + other_fn) *
#                                      (other_tn + other_fp) * (other_tn + other_fn)) ** 0.5
#         other_mcc = other_mcc_numerator / other_mcc_denominator if other_mcc_denominator > 0 else 0
#
#         pes_precision_list.append(target_precision)
#         pes_recall_list.append(target_recall)
#         # acc_list.append(acc.item())
#         pes_f1_list.append(target_f1_score)
#         pes_mcc_list.append(target_mcc)
#
#         other_precision_list.append(other_precision)
#         other_recall_list.append(other_recall)
#         other_f1_list.append(other_f1_score)
#         other_mcc_list.append(other_mcc)
#
# target_metrics = (np.mean(pes_f1_list), np.mean(pes_precision_list), np.mean(pes_recall_list),
#                       np.mean(pes_mcc_list))
# other_metrics = (np.mean(other_f1_list), np.mean(other_precision_list), np.mean(other_recall_list),
#                      np.mean(other_mcc_list))
#
# target_F1, target_preci, target_recall, target_mcc = target_metrics
# other_F1, other_preci, other_recall, other_mcc = other_metrics
#
# print("\n=== Target Base Pairs Results ===")
# print("test_preci=%.5f, test_recall=%.5f, test_F1=%.5f, test_MCC=%.5f" % (
#         target_preci, target_recall, target_F1, target_mcc))
#
# print("\n=== Other Base Pairs Results ===")
# print("test_preci=%.5f, test_recall=%.5f, test_F1=%.5f, test_MCC=%.5f" % (
#         other_preci, other_recall, other_F1, other_mcc))
#
