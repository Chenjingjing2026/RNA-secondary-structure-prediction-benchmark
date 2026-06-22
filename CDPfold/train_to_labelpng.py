#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:20:23 2018

@author: zhangch
"""

from __future__ import division
import os
import numpy as np
import csv
import math
import scipy.misc
import pandas as pd

from tqdm import tqdm
from PIL import Image
import numpy as np

            # 将原来的 scipy.misc.imsave 替换为 PIL.Image.save
def save_image(pic, new_filename):
    img = Image.fromarray(pic.astype(np.uint8))  # 转换为uint8类型，因为PIL需要这种格式
    img.save(new_filename)
    print(f"Saved image to {new_filename}")

def Gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y):
    if x == ['A'] and y == ['U']:
        return 2
    elif x == ['G'] and y == ['C']:
        return 3
    elif x == ["G"]and y == ['U']:
        return 0.8
    elif x == ['U'] and y == ['A']:
        return 2
    elif x == ['C'] and y == ['G']:
        return 3
    elif x == ["U"]and y == ['G']:
        return 0.8
    else:
        return 0

def readfile(path_dir,file):
    filename = path_dir + file
    csvFile = open(filename,"r")
    reader = csv.reader(csvFile)
    data = []
    for item in reader:
        data.append(item)
    return data,file

# def creatmat(sequence, structure):
#     mat = np.zeros([len(sequence), len(sequence)])
#     for i in range(len(sequence)):
#         for j in range(len(sequence)):
#             coefficient = 0
#             for add in range(30):
#                 if i - add >= 0 and j + add < len(sequence):
#                     score = paired(sequence[i - add], structure[j + add])
#                     if score == 0:
#                         break
#                     else:
#                         coefficient = coefficient + score * Gaussian(add)
#                 else:
#                     break
#             if coefficient > 0:
#                 for add in range(1, 30):
#                     if i + add < len(sequence) and j - add >= 0:
#                         score = paired(sequence[i + add], structure[j - add])
#                         if score == 0:
#                             break
#                         else:
#                             coefficient = coefficient + score * Gaussian(add)
#                     else:
#                         break
#             mat[i, j] = coefficient
#     return mat

#
def creatmat(data):
    mat = np.zeros([len(data),len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add <len(data):
                    score = paired(data[i - add],data[j + add])
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1,30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add],data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i],[j]] = coefficient
    return mat

def complete(i):
    if i < 10:
        str1 = '00' + str(i)
    elif i <100:
        str1 = '0' + str(i)
    else:
        str1 = str(i)
    return str1

def change(x):
    if x == ['(']:
        return 0
    elif x == [')']:
        return 1
    else:
        return 2

def jiaoyan(data):
    num_0 = num_1 =num_2 = 0
    for i in range(len(data)):
        if data[i][0] == '(':
            num_0 = num_0 + 1
        elif data[i][0] == ')':
            num_1 = num_1 + 1
        else:
            num_2 = num_2 + 1
        if num_1 == num_0:
            return True
        else:
            return False
    

# seq_dir = '/home/zhangch/RSS/5s/train_seq_data/'
# str_dir = '/home/zhangch/RSS/5s/train_stru_data/'

#
# seq_dir = '/home/zhangch/RSS/5s/train_seq_data/'
# str_dir = '/home/zhangch/RSS/5s/train_stru_data/'
png_dir = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/test_labelpng/'

# list_Dir = '/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/train_set/'
str_dir = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/stru_data/'
seq_dir = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/seq_data/'


if __name__ == "__main__":
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    pathDir = os.listdir(seq_dir)
    pathDir.sort()
    length = len(pathDir)
    for i in tqdm(range(length)):
        filename = pathDir[i]
        filename2 = filename[:-8] + '.csv'
        data, file = readfile(seq_dir, filename)
        data2, file2 = readfile(str_dir, filename2)
        if jiaoyan(data):
            im = np.zeros([len(data) + 19, len(data), 3])
            mat = creatmat(data)
            im[9:len(data) + 9, 0:len(data), 0] = im[9:len(data) + 9, 0:len(data), 0] + mat
            for j in range(len(data)):
                pic = im[j:j + 19]
                new_filename = png_dir + str(change(data2[j])) + '.' + file[:-4] + '_' + complete(j) + '.png'
                # scipy.misc.imsave(new_filename, pic)
                # 假设 pic 是一个numpy数组
                image = Image.fromarray(pic.astype(np.uint8))  # 将 numpy 数组转换为 PIL 图像对象
                image.save(new_filename)  # 使用 PIL 保存图像

        else:
            print(file)
                   
# def process_rna_data(parquet_file, output_dir):
#     # 读取Parquet文件
#     df = pd.read_parquet(parquet_file)
#
#     # 遍历Parquet文件中的每个RNA序列和结构
#     for idx, row in df.iterrows():
#         seq_id = row['id']  # 获取序列ID
#         sequence = row['sequence']  # 获取一级序列
#         structure = row['secondary_structure']  # 获取点括号结构
#     # pathDir =  os.listdir(seq_dir)
#     # pathDir.sort()
#     # length = len(pathDir)
#     # for i in range(length):
#     #     filename = pathDir[i]
#     #     filename2 = filename[:-8] + '.csv'
#     #     data,file = readfile(seq_dir , filename)
#     #     data2,file2 = readfile(str_dir,filename2)
#     #     if jiaoyan(data):
#     #         im = np.zeros([len(data)+19,len(data),3])
#     #         mat = creatmat(data)
#     #         im[9:len(data)+9,0:len(data),0] = im[9:len(data)+9,0:len(data),0] + mat
#     #         for j in range(len(data)):
#     #             pic = im[j:j+19]
#     #             new_filename=png_dir+str(change(data2[j]))+'.'+file[:-4] +'_'+complete(j) + '.png'
#     #             scipy.misc.imsave(new_filename, pic)
#     #     else:
#     #         print file
#             # 如果结构有效
#         if jiaoyan(structure):
#             # 创建图像矩阵
#             im = np.zeros([len(sequence) + 19, len(sequence), 3])
#             mat = creatmat(sequence, structure)
#             im[9:len(sequence) + 9, 0:len(sequence), 0] = im[9:len(sequence) + 9, 0:len(sequence), 0] + mat
#
#             # 保存图像
#             for j in range(len(sequence)):
#                 pic = im[j:j + 19]
#                 new_filename = os.path.join(output_dir, f"{seq_id}_{complete(j)}.png")
#                 save_image(pic, new_filename)
#
#                 print(f"Saved image for {seq_id} to {new_filename}")
#         else:
#             print(f"Invalid structure for {seq_id}")
#
# if __name__ == "__main__":
#     # 输入和输出路径
#     parquet_file = '/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/RNAStrAlign_short.parquet'
#     output_dir = '/home/chenjingjing/Models/CDPfold/data/RNAStrAlign/png/'
#
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 处理RNA数据并生成图像
#     process_rna_data(parquet_file, output_dir)
