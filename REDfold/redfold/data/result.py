import os
import torch
import math
from tqdm import tqdm

# def compare_bpseq(ref, pred):
#     L = len(ref) - 1
#     tp = fp = fn = tn =  0
#     if ((len(ref)>0 and isinstance(ref[0], list)) or (isinstance(ref, torch.Tensor) and ref.ndim==2)):
#         if isinstance(ref, torch.Tensor):
#             ref = ref.tolist()
#         ref = {(min(i, j), max(i, j)) for i, j in ref}
#         pred = {(i, j) for i, j in enumerate(pred) if i < j}
#         tp = len(ref & pred)
#         fp = len(pred - ref)
#         fn = len(ref - pred)
#     else:
#         # print("ref 的长度:", len(ref))
#         # print("pred 的长度:", len(pred))
#         assert(len(ref) == len(pred))
#         for i, (true_val, pred_val) in enumerate(zip(ref, pred)):
#             # 根据四种情况更新计数器
#             if pred_val > 0 and true_val > 0:
#                 # 都有配对
#                 if pred_val == true_val: # 配对正确
#                     tp += 1
#                 else: # 配对错误
#                     fn += 1
#             elif pred_val > 0 and true_val == 0:# 错误预测了配对
#                 fp += 1
#             elif pred_val == 0 and true_val > 0:# 漏掉了真实的配对
#                 fn += 1
#             else:  # pred_val == 0 and true_val == 0   # 正确预测无配对
#                 tn += 1
#     #
#     #         if j1 > 0 and i < j1: # pos
#     #             if j1 == j2:
#     #                 tp += 1
#     #             elif j2 > 0 and i < j2:
#     #                 fp += 1
#     #                 fn += 1
#     #             else:
#     #                 fn += 1
#     #         elif j2 > 0 and i < j2:
#     #             fp += 1
#     # tn = L * (L - 1) // 2 - tp - fp - fn
#     return tp, tn, fp, fn

def compare_bpseq(ref, pred):
    L = len(ref) - 1
    tp = fp = fn = 0
    if ((len(ref)>0 and isinstance(ref[0], list)) or (isinstance(ref, torch.Tensor) and ref.ndim==2)):
        if isinstance(ref, torch.Tensor):
            ref = ref.tolist()
        ref = {(min(i, j), max(i, j)) for i, j in ref}
        pred = {(i, j) for i, j in enumerate(pred) if i < j}
        tp = len(ref & pred)
        fp = len(pred - ref)
        fn = len(ref - pred)
    else:
        assert(len(ref) == len(pred))
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            if j1 > 0 and i < j1: # pos
                if j1 == j2:
                    tp += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            elif j2 > 0 and i < j2:
                fp += 1
    tn = L * (L - 1) // 2 - tp - fp - fn
    return (tp, tn, fp, fn)


def accuracy(tp, tn, fp, fn):
    sen = tp / (tp + fn) if tp+fn > 0. else 0.
    ppv = tp / (tp + fp) if tp+fp > 0. else 0.
    fval = 2 * sen * ppv / (sen + ppv) if sen+ppv > 0. else 0.
    mcc = ((tp*tn)-(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0. else 0.
    # 计算precision, recall, f1
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return sen, ppv, fval, mcc, precision, recall, f1
    # return (sen, ppv, fval, mcc)


import re
import numpy as np


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
        pairs = [int(line.split()[2]) for line in lines if len(line.split()) > 2]

        # 转换为 tensor 并存入字典
        ref[file_id] = torch.tensor(pairs)

    return ref

# 预测结果
file_path = '/home/chenjingjing/Models/REDfold1/redfold/data/pes/test.txt'  # replace with your file path
pred = parse_file(file_path)
for file_id, pair_tensor in pred.items():
    print(f"ID: {file_id}, Pair Tensor: {pair_tensor}")


# Ground truth
bpseq_folder = '/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/pseudoknot/short/test'  # 替换为你的 bpseq 文件夹路径
ref = parse_bpseq_files(bpseq_folder)

# 输出结果
# 计算字典中的键（key）数量
num_keys = len(ref)
sen_all = 0
ppv_all = 0
fval_all = 0
mcc_all = 0
precision_all = 0
recall_all = 0
f1_all = 0
# 用于存储每个样本的详细结果
detailed_results = []

for file_id in tqdm(ref.keys(), desc="Processing files"):
    # print(f"ID: {file_id}, Pair Tensor: {pair_tensor}")
    try:
        tp, tn, fp, fn = compare_bpseq(ref[file_id], pred[file_id])
    except KeyError:
        continue  # Skip to the next iteration if the key is not found
    # tp, tn, fp, fn = compare_bpseq(ref[file_id], pred[file_id])
    sen, ppv, fval, mcc, precision, recall, f1 = accuracy(tp, tn, fp, fn)

    sen_all += sen
    ppv_all += ppv
    fval_all += fval
    mcc_all += mcc
    precision_all += precision
    recall_all += recall
    f1_all += f1

    # 记录每个样本的详细指标 - 表格格式，空格分隔
    detailed_results.append(f"{file_id} {precision:.4f} {recall:.4f} {f1:.4f} {mcc:.4f}")

fn_sen = sen_all/num_keys
fn_ppv = ppv_all/num_keys
fn_fval = fval_all/num_keys
fn_mcc = mcc_all/num_keys
fn_precision = precision_all/num_keys
fn_recall = recall_all/num_keys
fn_f1 = f1_all/num_keys

# 打开文件并以续写的方式写入数据
with open(file_path, 'a') as file:
    file.write(f"\n------------------result------------------\n")
    file.write(f"fn_sen: {fn_sen}\n")
    file.write(f"fn_ppv: {fn_ppv}\n")
    file.write(f"fn_fval: {fn_fval}\n")
    file.write(f"fn_mcc: {fn_mcc}\n")
    file.write(f"fn_precision: {fn_precision}\n")
    file.write(f"fn_recall: {fn_recall}\n")
    file.write(f"fn_f1: {fn_f1}\n")
    file.write("\n")  # 在数据末尾加一个空行

# 写入每个样本的详细指标到另一个txt文件
output_file = '/home/chenjingjing/Models/REDfold1/redfold/data/pes/detailed_metrics.txt'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for result in detailed_results:
        f.write(result + '\n')

print(f"\n详细指标已保存到: {output_file}")