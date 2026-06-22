import os
from tqdm import tqdm  # 导入 tqdm

# 输入和输出文件夹路径
bpseq_folder = '/home/chenjingjing/DATA/data/Crossfamily_new/tRNA/test'  # 替换为你的 bpseq 文件夹路径
fasta_folder = '/home/chenjingjing/Models/REDfold1/redfold/data/crossfamily_short/tRNA/test'  # 替换为你希望保存 fasta 文件的文件夹路径

# 确保输出文件夹存在
os.makedirs(fasta_folder, exist_ok=True)

# 获取 bpseq 文件夹中的所有文件
bpseq_files = [f for f in os.listdir(bpseq_folder) if f.endswith('.bpseq')]

# 使用 tqdm 显示进度条
for filename in tqdm(bpseq_files, desc="Converting files", unit="file"):
        # 构造文件路径
        bpseq_file_path = os.path.join(bpseq_folder, filename)

        # 打开并读取 bpseq 文件
        with open(bpseq_file_path, 'r') as bpseq_file:
            lines = bpseq_file.readlines()

        # 初始化一个空的列表来存储每一行的第二列数据
        sequences = []

        # 遍历每一行，提取第二列
        for line in lines:
            columns = line.strip().split()  # 按空格或制表符分割行
            if len(columns) > 1:  # 确保这一行有至少两列
                sequences.append(columns[1])  # 获取第二列

        # 将所有第二列数据合并成一个长的序列（如果需要）
        sequence = ''.join(sequences)

        # 构造fasta文件名和路径
        fasta_file_path = os.path.join(fasta_folder, filename.replace('.bpseq', '.fasta'))

        # 写入fasta文件
        with open(fasta_file_path, 'w') as fasta_file:
            fasta_file.write(f">{filename.replace('.bpseq', '')}\n")  # >后面是id，文件名去掉扩展名
            fasta_file.write(sequence + '\n')

print("转换完成！")
