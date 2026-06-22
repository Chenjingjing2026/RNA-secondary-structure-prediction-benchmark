# import os
# import numpy as np
# from tqdm import tqdm  # 导入 tqdm 库
#
# def one_hot_encode(base):
#     """将单个碱基转换为one-hot编码"""
#     encoding = {
#         'A': [1, 0, 0, 0],
#         'C': [0, 1, 0, 0],
#         'G': [0, 0, 1, 0],
#         'U': [0, 0, 0, 1]
#     }
#     return encoding.get(base, [0, 0, 0, 0])
#
#
# def process_file(folder_path, file_name, output_path):
#     """处理单个文件，生成序列文件和结构文件"""
#     file_path = os.path.join(folder_path, file_name)
#     sequence = []
#     structure = []
#
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 3:
#                 _, base, struct = parts
#                 sequence.extend(one_hot_encode(base))
#                 structure.append(struct)
#
#     # 生成序列文件
#     seq_file_name = file_name.replace('.bpseq', '_sequence.txt')
#     seq_file_path = os.path.join(output_path, seq_file_name)
#     with open(seq_file_path, 'w') as seq_file:
#         seq_file.write(','.join(map(str, sequence)) + '\n')
#
#     # 生成结构文件
#     struct_file_name = file_name.replace('.bpseq', '_structure.txt')
#     struct_file_path = os.path.join(output_path, struct_file_name)
#     with open(struct_file_path, 'w') as struct_file:
#         struct_file.write(','.join(structure) + '\n')
#
#     return file_name[:-6], seq_file_name, struct_file_name
#
#
# def main(folder_path, output_path):
#     """主函数，处理文件夹中的所有文件"""
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     print(output_path)
#     file_list_path = os.path.join(output_path, 'test.txt')
#     # file_list_path = os.path.join(output_path, 'filelist')
#     # file_list_path= output_path
#     # with open(file_list_path, 'w') as file_list:
#     #     for file_name in os.listdir(folder_path):
#     #         if file_name.endswith('.bpseq') and not file_name.startswith('filelist'):
#     #             seq_name, seq_file_name, struct_file_name = process_file(folder_path, file_name, output_path)
#     #             file_list.write(f"{seq_name}\t{seq_file_name}\t{struct_file_name}\n")
#     # 获取文件夹中的所有文件
#     files = [file_name for file_name in os.listdir(folder_path) if
#              file_name.endswith('.bpseq') and not file_name.startswith('filelist')]
#
#     # 使用 tqdm 显示进度条
#     with open(file_list_path, 'w') as file_list:
#         # 使用 tqdm 包装迭代器来显示进度条
#         for file_name in tqdm(files, desc="Processing files", unit="file"):
#             seq_name, seq_file_name, struct_file_name = process_file(folder_path, file_name, output_path)
#             file_list.write(f"{seq_name}\t{seq_file_name}\t{struct_file_name}\n")
#
# # 示例用法
# # folder_path = input("请输入文件夹路径：")
# main('/home/chenjingjing/DATA/data/Crossfamily/16s/train/', '/home/chenjingjing/Models/SPOT-RNA/SPOT-RNA-DL/mydata/crossfamily_new/16s/train/')

import os
import numpy as np
from tqdm import tqdm  # 导入进度条库


def one_hot_encode(base):
    """将单个碱基转换为one-hot编码"""
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1]
    }
    return encoding.get(base, [0, 0, 0, 0])  # 未知碱基返回全零


def process_file(folder_path, file_name, output_path, max_length):
    """
    处理单个.bpseq文件，生成序列和结构文件
    参数:
        folder_path: 输入文件夹路径
        file_name: 文件名
        output_path: 输出文件夹路径
        max_length: 允许的最大序列长度
    返回:
        成功时返回 (序列名, 序列文件名, 结构文件名)
        长度超标时返回 None
    """
    file_path = os.path.join(folder_path, file_name)
    sequence = []  # 存储one-hot编码序列
    structure = []  # 存储二级结构
    current_length = 0  # 当前序列长度计数器

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # 确保是有效行
                _, base, struct = parts
                sequence.extend(one_hot_encode(base))
                structure.append(struct)
                current_length += 1

                # 实时检查长度限制
                if current_length > max_length:
                    return None  # 超过长度立即终止处理

    # 保存序列文件(one-hot编码)
    seq_file_name = file_name.replace('.bpseq', '_sequence.txt')
    seq_file_path = os.path.join(output_path, seq_file_name)
    with open(seq_file_path, 'w') as seq_file:
        seq_file.write(','.join(map(str, sequence)) + '\n')

    # 保存结构文件
    struct_file_name = file_name.replace('.bpseq', '_structure.txt')
    struct_file_path = os.path.join(output_path, struct_file_name)
    with open(struct_file_path, 'w') as struct_file:
        struct_file.write(','.join(structure) + '\n')

    return file_name[:-6], seq_file_name, struct_file_name


def main(folder_path, output_path, max_length,filename):
    """
    主处理函数，批量处理文件夹中的所有.bpseq文件
    参数:
        folder_path: 输入文件夹路径
        output_path: 输出文件夹路径
        max_length: 序列最大允许长度(默认500)
    """
    # 创建输出目录(如果不存在)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"输出目录: {output_path}")
    file_list_path = os.path.join(output_path, filename)  # 文件列表路径

    # 筛选所有.bpseq文件(排除filelist文件)
    files = [f for f in os.listdir(folder_path)
             if f.endswith('.bpseq') and not f.startswith('filelist')]

    # 带进度条的批量处理
    with open(file_list_path, 'w') as file_list:
        for file_name in tqdm(files, desc="正在处理文件", unit="个"):
            result = process_file(folder_path, file_name, output_path, max_length)

            # 只记录成功处理的文件(长度符合要求)
            if result is not None:
                seq_name, seq_file, struct_file = result
                file_list.write(f"{seq_name}\t{seq_file}\t{struct_file}\n")

    print(f"处理完成! 已过滤长度超过{max_length}的序列。")


if __name__ == "__main__":
    datasets = ['16s','5s','grp1','grp2','RNaseP','srp','telomerase','tRNA','tmRNA']
    max_len = 1024
    for dataset in datasets:
        train_input_folder = f'/home/cjj/Model_Data/Crossfamily/{dataset}/train/'
        train_output_folder = f"/home/cjj/Spot-RNA/SPOT-RNA-DL/data/Crossfamily_{max_len}/{dataset}/train/"
        main(train_input_folder, train_output_folder, max_len,'train.txt')
        valid_input_folder = f'/home/cjj/Model_Data/Crossfamily/{dataset}/valid/'
        valid_output_folder = f"/home/cjj/Spot-RNA/SPOT-RNA-DL/data/Crossfamily_{max_len}/{dataset}/valid/"
        main(valid_input_folder, valid_output_folder, max_len,'valid.txt')
        test_input_folder = f'/home/cjj/Model_Data/Crossfamily/{dataset}/test/'
        test_output_folder = f"/home/cjj/Spot-RNA/SPOT-RNA-DL/data/Crossfamily_{max_len}/{dataset}/test/"
        
        main(test_input_folder, test_output_folder, max_len,'test.txt')
    # 调用主函数(可调整max_length参数)
    



