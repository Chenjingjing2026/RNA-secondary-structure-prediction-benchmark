import os
from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from queue import Queue, LifoQueue, PriorityQueue


# # 返回 RNA 二级结构二维数组，配对的标255，不配对的标0
# def encode_with_pairing_matrix_from_bpseq(length, width, bpseq_data):
#     """
#     根据 bpseq 数据生成对应的配对图像矩阵。
#     配对的碱基标255，不配对的标0。
#     """
#     matrix = [([0] * length) for i in range(width)]  # 初始化全0的矩阵
#
#     # 创建一个字典来记录配对关系
#     dict = {}
#
#     # 遍历 bpseq 数据，获取配对信息
#     for line in bpseq_data:
#         listFromLine = line.strip().split()
#         five = int(listFromLine[0])  # 当前碱基的位置
#         four = int(listFromLine[2])  # 配对碱基的位置
#
#         if four != 0:  # 如果有配对关系
#             dict[five] = four  # 当前碱基与配对碱基建立关系
#
#     # 遍历配对关系，填充矩阵
#     for key, value in dict.items():
#         # 将配对的碱基标记为255
#         matrix[key // width][key % length] = 255
#         matrix[value // width][value % length] = 255
#
#     print('encoded pairing matrix:', matrix)
#     return matrix

# 返回 RNA 二级结构一维数组
def helper4(path,lab_path,function):
    result = []
    files = []
    B =[]
    with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/' + lab_path + '/' + f'{function}.txt') as f:
    # with open('..//lable//'+lab_path+'//'+lab_path+'.txt') as f:
        for line in f.readlines():
            a = line.strip('\n')
            a = a.split("*")
            # b = a[0].rstrip('.ct')
            b = a[0].rstrip('.bpseq')
            B.append(b)
            fname = path + a[0]
            files.append(fname)
        for file in files:
            fr = open(file)
            # length = fr.readline().split('\t')[0]  # 获取序列长度
            length = sum(1 for _ in fr)
            print('length',length)# Count lines in the file for first dimension
            dict = {}
            index = 15
            res = [0] * int(length)
            flags = [0] * int(length)
            # 读取 bpseq 文件内容
            # bpseq_data = fr.readlines()
            #
            # # 生成配对图像矩阵
            # arr = encode_with_pairing_matrix_from_bpseq(24, 24, bpseq_data)  # 使用修改后的配对函数
            #
            # result.append(res)
            for line in fr.readlines():
                line = line.strip()  # 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                # listFromLine = line.split('\t')  # \t表示空四个字符，也称缩进，相当于按一下Tab键
                listFromLine = line.split()  # \t表示空四个字符，也称缩进，相当于按一下Tab键
                # four = int(listFromLine[4])
                four = int(listFromLine[2])
                five = int(listFromLine[0])
                if (four != 0 and flags[five - 1] == 0):
                    dict[five] = four
                if (four == 0 and len(dict) != 0):
                    # print(dict)
                    for key, value in dict.items():
                        if (res[int(key) - 1] == 0 and res[int(value) - 1] == 0):
                            res[int(key) - 1] = index
                            res[int(value) - 1] = index
                            flags[int(key) - 1] = 1
                            flags[int(value) - 1] = 1
                    index = index+10
                    dict.clear()
            result.append(res)
        # print('result,B',result,B)
        return result,B


# 返回 RNA 二级结构二维数组
def helper5(length, width, arr=[]):
    matrix = [([255] * length) for i in range(width)]
    q = Queue(maxsize=0)  # 创建队列
    for x in arr:
        q.put(x)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (q.qsize() == 0):
                break
            matrix[i][j] = q.get()
    # print('helper5 matrix',matrix)
    return matrix


def helper51(length, width, list=[]): #正反
    matrix = [([255] * length) for i in range(width)]
    # print(matrix)
    q = Queue(maxsize=0)  # 创建队列
    list = list[0:(length * width)]  # 保留length*width个元素
    # 编码RNA序列
    # A->0，G->85，C->170，U->255
    list_temp1 = []
    list_temp2 = []
    res = []
    for i in range(width):
        if i % 2 == 0:  # 偶数
            res = res + list[i * width:(i + 1) * width]
        if i % 2 != 0:  # 奇数
            list_temp1 = list[i * width:(i + 1) * width]
            for i in range(0, len(list_temp1)):
                list_temp2.insert(0, list_temp1[i])
            res = res + list_temp2
            list_temp1.clear()
            list_temp2.clear()

    for x in res:
        q.put(x)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            # print(len(matrix[i]))
            if (q.qsize() == 0):
                break
            matrix[i][j] = q.get()
    # print('helper51 matrix', matrix)
    return matrix



# 制作RNA二级结构图像
def helper6(img, index,save_path):
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        img.flags.writeable = True
        pic = Image.fromarray(np.uint8(img))
        pic.save(os.path.join(save_path, f"{index}-f.png"))
    except Exception as e:
        print(f"Error saving image for index {index}: {e}")
    # img.flags.writeable = True  # 将数组改为读写模式
    # pic = Image.fromarray(np.uint8(img))
    # pic.save(save_path + str(index) + '-f.png')

def helper61(img, index,save_path):
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Proceed with saving the image
    try:
        img.flags.writeable = True
        pic = Image.fromarray(np.uint8(img))
        pic.save(os.path.join(save_path, f"{index}-fb.png"))
    except Exception as e:
        print(f"Error saving image for index {index}: {e}")
    #
    # img.flags.writeable = True  # 将数组改为读写模式
    # pic = Image.fromarray(np.uint8(img))
    # pic.save(save_path + str(index) + '-fb.png')

def main():
    name = 'bprna_new'
    function='test' #test train val
    # '/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/train_set'
    # '/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/valid_set'
    # '/home/chenjingjing/DATA/data/archiveII/shortdata/bpseq/'
    data,rname = helper4('/home/chenjingjing/DATA/data/bprna-new/without_pes/short/bpseq/',name,function)
    index = 1
    for x,y in zip(data,rname):
        # print(index,len(x))
        img = helper5(24,24,x)
        img1 = helper51(24,24,x)
        # sive_path = "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + name + "/train/seq_photo/"
        helper6(np.array(img),y,"/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+name+f'/{function}/str_photo/')
        helper61(np.array(img1),y,"/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+name+f'/{function}/str_photo/')
        index = index + 1

if __name__ == "__main__":
    main()