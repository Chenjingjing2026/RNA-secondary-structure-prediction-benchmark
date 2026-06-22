import os
from tqdm import tqdm  # 导入 tqdm

def extract_unique_file_names(folder_path, output_file):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)

    # 使用一个 set 来存储唯一的文件名
    unique_names = set()

    # 打开输出txt文件
    with open(output_file, 'w') as f:
        # 使用 tqdm 显示进度条，设置 total 为文件的总数
        for file in tqdm(files, desc="Processing files", unit="file"):
            if file.endswith('.png'):  # 只处理 png 文件
                # 找到 '-f' 或 '-fb' 之前的部分
                if '-f' in file:
                    name_part = file.split('-f')[0]
                elif '-fb' in file:
                    name_part = file.split('-fb')[0]
                else:
                    # 如果文件名没有 '-f' 或 '-fb'，就直接取文件名（去掉扩展名）
                    name_part = os.path.splitext(file)[0]

                # 将文件名加入 set（去重）
                unique_names.add(name_part)

        # 将去重后的文件名写入 txt 文件
        for name in unique_names:
            f.write(name + '.bpseq'+ '\n')
            # print(f"写入: {name}")

# 使用示例
folder_path = '/home/chenjingjing/Models/MSFF-CDCGAN/picture/RNAStrAlign/val/seq_photo'  # 这里换成你文件夹的路径
output_file = '/home/chenjingjing/Models/MSFF-CDCGAN/lable/RNAStrAlign/val/val-De-redundancy.txt'  # 输出的txt文件路径
extract_unique_file_names(folder_path, output_file)
