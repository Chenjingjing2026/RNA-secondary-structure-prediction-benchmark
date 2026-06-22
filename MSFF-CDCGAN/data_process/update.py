import os

# 输入txt文件路径
input_file = '/home/chenjingjing/Models/MSFF-CDCGAN/lable/bprna_new/test/test-data.txt'
output_file = '/home/chenjingjing/Models/MSFF-CDCGAN/lable/bprna_new/test/Filtered_test-data.txt'

# 打开原文件和输出文件
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # 切割每一行
        paths = line.split('*')
        if len(paths) < 2:
            continue

        # 获取文件路径
        path1 = paths[0]
        path2 = paths[1]

        # 检查文件是否存在
        if os.path.exists(path1) and os.path.exists(path2):
            outfile.write(line)  # 如果路径存在，保留这一行

print("清理完成，已保存到", output_file)
