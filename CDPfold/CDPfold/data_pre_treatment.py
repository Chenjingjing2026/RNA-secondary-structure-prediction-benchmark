#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:36:01 2018

@author: zhangch
"""
import os
import pandas as pd
import csv
from tqdm import tqdm

 # 遍历指定目录，显示目录下的所有文件名
def readfile(fileDir,filename):
    fopen = open(fileDir+filename,'r')
    rnafile = fopen.readlines()
    # del rnafile[0]
    with open("train.txt",'w') as f:
        for i in rnafile:
            f.write(i)
    data = pd.read_csv("train.txt", sep=' ', header=None)
    return data,filename

def transform(data,filename):
    # rnaseq = data.loc[:,1]
    # rnadata1 = data.loc[:,0]
    # rnadata2 = data.loc[:,2]
    # print(data.head())  # 打印前几行检查数据
    # print(data.columns)  # Check the column names
    # print(data.shape)  # Check the number of rows and columns

    rnaseq = data.iloc[:, 1]  # Access the sequence column (second column)
    rnadata1 = data.iloc[:, 0]  # Access the row number column (first column)
    rnadata2 = data.iloc[:, 2]  # Access the numeric value column (third column)
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] == 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return rnaseq,rnastructure,filename

def savefile(rnaseq,rnastructure,filename):
    rnafile = filename[:-3]
    rnafile = rnafile+".csv"
    rnafile = csv_Dir + rnafile
    if not os.path.exists(csv_Dir):
        os.makedirs(csv_Dir)
    rnacsv = open(rnafile,'w')
    writer = csv.writer(rnacsv)
    m = len(rnastructure)
    for i in range(m):
        writer.writerow(rnastructure[i])
    os.remove("train.txt")
    rnaseqfile = seq_Dir + filename[:-3] + "_seq.csv"
    if not os.path.exists(seq_Dir):
        os.makedirs(seq_Dir)
    rnaseqcsv = open(rnaseqfile,'w')
    seqwriter = csv.writer(rnaseqcsv)
    m = len(rnastructure)
    for i in range(m):
        seqwriter.writerow(rnaseq[i])

list_Dir = '/home/chenjingjing/DATA/data/bprna-new/without_pes/short/bpseq/'
# list_Dir = '/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/valid_set/'
# list_Dir = '/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/train_set/'
csv_Dir = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/stru_data/'
seq_Dir = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/seq_data/'

# list_Dir = '/home/zhangch/RSS/source_data/'
# csv_Dir = '/home/zhangch/RSS/stru_data/'
# seq_Dir = '/home/zhangch/RSS/seq_data/'

if __name__ == '__main__':
    pathDir =  os.listdir(list_Dir)
    # for i in pathDir:
    for i in tqdm(pathDir, desc="Processing files"):
        data,filename=readfile(list_Dir , i)
        rnaseq,rnastructure,filename=transform(data,filename)
        savefile(rnaseq,rnastructure,filename)
    

    
