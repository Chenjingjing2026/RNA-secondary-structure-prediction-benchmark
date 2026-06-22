# 读取文件
with open('/home/chenjingjing/Models/DMfold/Saver_Result/bprna_new/results.txt', 'r') as file:
    lines = file.readlines()

# 初始化求和变量
sum_count = 0
sum_sen = 0
sum_ppv = 0
sum_fscore = 0
sum_mcc = 0

# 计算每列的和
for line in lines:
    values = line.strip().split(',')
    if len(values) == 6:  # 确保这一行有6个值
        sum_count += float(values[0])
        sum_sen += float(values[1])
        sum_ppv += float(values[2])
        sum_fscore += float(values[4]) if values[4] else 0  # 防止空值
        sum_mcc += float(values[5])

# 计算平均值
count = len(lines)  # 行数
average_count = sum_count / count
average_sen = sum_sen / count
average_ppv = sum_ppv / count
average_fscore = sum_fscore / count
average_mcc = sum_mcc / count

# 写回文件
with open('/home/chenjingjing/Models/DMfold/Saver_Result/bprna_new/results.txt', 'a') as file:
    # file.write("{}, {}, ,{}, {}\n".format(
    #     round(average_sen, 6),
    #     round(average_ppv, 6),
    #     round(average_fscore, 6),
    #     round(average_mcc, 6)
    # ))
    file.write("\n####################################\n")

    # file.write("平均值：Count = {}\n".format(round(average_count, 6)))
    file.write("平均值：Sen = {}\n".format(round(average_sen, 6)))
    file.write("平均值：Ppv = {}\n".format(round(average_ppv, 6)))
    file.write("平均值：Fscore = {}\n".format(round(average_fscore, 6)))
    file.write("平均值：MCC = {}\n".format(round(average_mcc, 6)))

print("计算完毕并已写入文件。")
