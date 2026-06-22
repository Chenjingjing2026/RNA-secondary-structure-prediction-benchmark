import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库

def one_hot_encode(base):
    """将单个碱基转换为one-hot编码"""
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1]
    }
    return encoding.get(base, [0, 0, 0, 0])


def process_file(folder_path, file_name, output_path):
    """处理单个文件，生成序列文件和结构文件"""
    file_path = os.path.join(folder_path, file_name)
    sequence = []
    structure = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                _, base, struct = parts
                sequence.extend(one_hot_encode(base))
                structure.append(struct)

    # 生成序列文件
    seq_file_name = file_name.replace('.bpseq', '_sequence.txt')
    seq_file_path = os.path.join(output_path, seq_file_name)
    with open(seq_file_path, 'w') as seq_file:
        seq_file.write(','.join(map(str, sequence)) + '\n')

    # 生成结构文件
    struct_file_name = file_name.replace('.bpseq', '_structure.txt')
    struct_file_path = os.path.join(output_path, struct_file_name)
    with open(struct_file_path, 'w') as struct_file:
        struct_file.write(','.join(structure) + '\n')

    return file_name[:-6], seq_file_name, struct_file_name


def main(folder_path, output_path,name):
    """主函数，处理文件夹中的所有文件"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(output_path)
    file_list_path = os.path.join(output_path, f'{name}.txt')
    # file_list_path = os.path.join(output_path, 'filelist')
    # file_list_path= output_path
    # with open(file_list_path, 'w') as file_list:
    #     for file_name in os.listdir(folder_path):
    #         if file_name.endswith('.bpseq') and not file_name.startswith('filelist'):
    #             seq_name, seq_file_name, struct_file_name = process_file(folder_path, file_name, output_path)
    #             file_list.write(f"{seq_name}\t{seq_file_name}\t{struct_file_name}\n")
    # 获取文件夹中的所有文件
    files = [file_name for file_name in os.listdir(folder_path) if
             file_name.endswith('.bpseq') and not file_name.startswith('filelist')]
    print(len(files))

    # 使用 tqdm 显示进度条
    with open(file_list_path, 'w') as file_list:
        # 使用 tqdm 包装迭代器来显示进度条
        for file_name in tqdm(files, desc="Processing files", unit="file"):
            seq_name, seq_file_name, struct_file_name = process_file(folder_path, file_name, output_path)
            file_list.write(f"{seq_name}\t{seq_file_name}\t{struct_file_name}\n")

# 示例用法
# folder_path = input("请输入文件夹路径：")
name = 'train'
main(f'/root/lanyun-tmp/SPOT-RNA-DL/mydata/telomerase/{name}/', f'/root/lanyun-tmp/SPOT-RNA-DL/data/telomerase/{name}',name)
