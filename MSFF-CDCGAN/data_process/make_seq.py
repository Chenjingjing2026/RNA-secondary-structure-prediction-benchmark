import os
from numpy import *
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from queue import Queue, LifoQueue, PriorityQueue

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# 读取data_set文件夹下的.tc文件，取出RNA序列数据放入res中,并返回res
def Read_File(path,lab_path):
    res = []
    files=[]
    B = []
    # with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/' + lab_path + '/' + 'train.txt') as f:
    # with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/'+lab_path+'//'+lab_path+'.txt') as f:
    with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/bprna_new/' + lab_path + '.txt') as f:
    # output_lst_file = '/home/chenjingjing/Models/MSFF-CDCGAN/lable/Rivas/TrainSetA.txt'
        for line in f.readlines():
            a = line.strip('\n')
            a = a.split("*")
            b = a[0].rstrip('.bpseq')
            # b = a[0].rstrip('.ct')
            B.append(b)
            fname = path + a[0]
            files.append(fname)
        for file in files:
            fr = open(file)
            # next(fr)  # 跳过第一行，从第二行开始读取
            num = []
            for line in fr.readlines():
                line = line.strip()  # 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                # listFromLine = line.split('\t') # \t表示空四个字符，也称缩进，相当于按一下Tab键
                listFromLine = line.split()
                num.extend(listFromLine[1])
            res.append(num)
        # 将 res 中的所有数据转换为大写
        res = [[x.upper() for x in sublist] for sublist in res]  # 对 res 列表中的每个子列表做大写转换
    # print('res,B',res,B)
    return res,B


# 正序把RNA序列编码成图像（length * width）中的像素点
def encode(length, width, list=[]):
    num = int(ceil(length * width / len(list)))  # ceil函数实现上取整
    temp = list
    for i in range(num - 1):
        list = list + temp
    list = list[0:(length * width)]  # 保留length*width个元素
    # 编码RNA序列
    # A->0，G->85，C->170，U->255
    for index, value in enumerate(list):
        if (value == 'A'):
            list[index] = 0
        elif (value == 'G'):
            list[index] = 85
        elif (value == 'C'):
            list[index] = 170
        elif (value == 'U'):
            list[index] = 255
        else:
            # 如果遇到非法字符，直接返回None或退出
            return None
    arr = np.array(list)  # 把list转化为数组
    arr = arr.reshape(width, length)  # 改变数组的形状
    # 返回数组类型
    # print('encoude arr', arr)
    return arr


# 正序+逆序把RNA序列编码成图像（length * width）中的像素点
def encode2(length, width, list=[]):
    num = int(ceil(length * width / len(list)))  # ceil函数实现上取整
    temp = list
    for i in range(num - 1):
        list = list + temp
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
    for index, value in enumerate(res):
        if (value == 'A'):
            list[index] = 0
        elif (value == 'G'):
            list[index] = 85
        elif (value == 'C'):
            list[index] = 170
        elif (value == 'U'):
            list[index] = 255
        else:
            # 如果遇到非法字符，直接返回None或退出
            return None
    arr = np.array(list)  # 把list转化为数组
    arr = arr.reshape(width, length)  # 改变数组的形状
    # 返回数组类型
    # print('encoude2 arr', arr)
    return arr

# 制作RNA单序列二维矩阵，255/0填充空白
def encode3(length, width, list=[]):
    matrix = [([255] * length) for i in range(width)]
    # matrix = torch.randn([length, width])
    # 编码RNA序列
    for index, value in enumerate(list):
        if (value == 'A'):
            list[index] = 0
        elif (value == 'G'):
            list[index] = 1
        elif (value == 'C'):
            list[index] = 2
        elif (value == 'U'):
            list[index] = 3
        else:
            # 如果遇到非法字符，直接返回None或退出
            return None
    index = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if(index>=len(list)):
                break
            matrix[i][j] = list[index]
            index = index + 1
    # print('encoude3 matrix',matrix)
    return matrix

# 单序列正序+逆序把RNA序列编码成图像（length * width）中的像素点
def encode4(length, width, list=[]):
    matrix = [([255] * length) for i in range(width)]
    for index, value in enumerate(list):
        if (value == 'A'):
            list[index] = 0
        elif (value == 'G'):
            list[index] = 1
        elif (value == 'C'):
            list[index] = 2
        elif (value == 'U'):
            list[index] = 3
        else:
            # 如果遇到非法字符，直接返回None或退出
            return None
    list_temp = []
    res = []
    for i in range(width):
        if i %2 == 0:  # 偶数
            res = res + list[i * width:(i + 1) * width]
        if i % 2 != 0:  # 奇数
            list_temp1 = list[i * width:(i + 1) * width]
            for i in range(0, len(list_temp1)):
                list_temp.insert(0, list_temp1[i])
            res = res + list_temp
            list_temp1.clear()
            list_temp.clear()
    index = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (index >= len(res)):
                break
            matrix[i][j] = res[index]
            index = index + 1
    # print('encoude4 matrix',matrix)
    return matrix

# 制作灰度图像
def grayImage(img, index, sive_path):
    if img is not None:
        img.flags.writeable = True  # 将数组改为读写模式
        pic = Image.fromarray(np.uint8(img))
        pic.save(sive_path + index + '-f.png')


def grayImage2(img, index, sive_path):
    if img is not None:
        img.flags.writeable = True  # 将数组改为读写模式
        pic = Image.fromarray(np.uint8(img))
        pic.save(sive_path + index + '-fb.png')


def main():
    name = 'test'
    datasets,RNAname = Read_File('/home/chenjingjing/DATA/data/bprna-new/without_pes/short/bpseq/',name)
    # datasets,RNAname = Read_File("..//data//"+name+"//",name)
    index = 1
    sive_path = "/home/chenjingjing/Models/MSFF-CDCGAN/picture/bprna_new/"+name+"/seq_photo/"
    print(sive_path)
    if not os.path.exists(sive_path):
        os.makedirs(sive_path)
    for x,y in zip(datasets,RNAname):
        x1 = x.copy()
        y1 = y
        img = encode(24, 24, x)
        img1 = encode2(24, 24, x1)
        # grayImage(np.array(img),y,sive_path)
        # grayImage2(np.array(img1),y1,sive_path)
        # Only call grayImage and grayImage2 if images are valid
        if img is not None:
            grayImage(np.array(img), y, sive_path)
        else:
            print(f"Skipping grayImage for RNA {y} due to invalid img")

        if img1 is not None:
            grayImage2(np.array(img1), y, sive_path)
        else:
            print(f"Skipping grayImage2 for RNA {y} due to invalid img1")

        index = index + 1



if __name__ == "__main__":
    main()
