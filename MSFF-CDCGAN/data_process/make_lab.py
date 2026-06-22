from numpy import *
import numpy as np
import os
from numpy import *
import os
import imghdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from queue import Queue, LifoQueue, PriorityQueue

# 第一种方法
# index = 1
# with open("label_test.txt", "r") as f:
#     for line in f.readlines():
#         word = line.split("-")
#         a = word[0]
#         b = word[1]
#         c = word[new]
#         num = a.split("/")[new]
#         print(index)
#         index = index + 1
#         with open("label_3.txt", "a") as file:
#             file.write("SeqImg_2/" + str(num) + "-" + "StrImg/"+ num + "-" + c)
#
# f.close()  # 关闭文件

lable = {'5s':1,'16s':2,'23s':3,'grp1':4,'grp2':5,'RNaseP':6,'srp':7,'tmRNA':8,'tRNA':9,'telomerase':10,'5S':1,'16S':2,'23S':3,'group_I_intron':4,'group_II_intron':5,'SRP':7}
# 基础标签
def Read_File(path,data_path,function):
    files = os.listdir(data_path)
    with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/' + path + '/' + function + '.txt', 'r') as fd:
    # with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/'+path+'/'+function+'/'+function+'-De-redundancy.txt','r') as fd:
    # with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/'+path+'/'+path+'-De-redundancy.txt','r') as fd:
        for lines in fd.readlines():
            line=lines.strip()
            if line in files:
                fname = data_path + line
                with open(fname,'r') as fr:
                    # 计算文件行数
                    num_lines = sum(1 for _ in fr)
                    # 获取文件名（去除路径）并去掉后缀
                    file_name = os.path.basename(fname)  # 获取文件名
                    file_name_without_extension = os.path.splitext(file_name)[0]  # 去掉文件后缀
                    # 创建 first_line 列表
                    first_line = [num_lines, file_name_without_extension]
                    print('length', first_line)
                    # lines = fr.readlines()
                    # first_line = lines[0]
                    # first_line = first_line.split('\t')
                    # print('first_line',first_line)
                with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/' + path + '/' + function + '/' + function +'.txt','a') as f:
                    for key, value in lable.items():
                        if key in line:
                            f.write(line + '*' + str(first_line[0]) + '*' + str(value) + '\n')
                        else:
                            f.write(line + '*' + str(first_line[0]) + '*' + str(0) + '\n')

                            # f.write(line + '*' + first_line[0]+'*'+str(value)+'\n')



#  数据标签
def Read_File2(path,data_path,function):
    files = os.listdir(data_path)
    # with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/'+path+'/'+function+'/'+function+'-De-redundancy.txt','r') as fd:
    with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/' + path + '/' + function + '.txt','r') as fd:
        for lines in fd.readlines():
            line=lines.strip() #去冗余名字
            if line in files:
                fname = data_path + line
                with open(fname,'r') as fr:
                    # lines = fr.readlines()
                    # first_line = lines[0]
                    # first_line = first_line.split('\t') #取出对应去冗余数据的序列长度
                    # 计算文件行数
                    num_lines = sum(1 for _ in fr)
                    # 获取文件名（去除路径）并去掉后缀
                    file_name = os.path.basename(fname)  # 获取文件名
                    file_name_without_extension = os.path.splitext(file_name)[0]  # 去掉文件后缀
                    # 创建 first_line 列表
                    first_line = [num_lines, file_name_without_extension]
                    print('length', first_line)

                with open('/home/chenjingjing/Models/MSFF-CDCGAN/lable/' + path + '/' + function + '/' + function +'-data.txt','a') as f:
                    # line = line.rstrip('.ct')
                    line = line.rstrip('.bpseq')
                    for key, value in lable.items():
                        if key in line:
                            f.write("/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/seq_photo/" +str(line) +'-f.png'+ "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/str_photo/" + str(line)+ "-f.png" + "*" + str(first_line[0])+ "*" +str(value)+ '\n')
                            f.write("/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/seq_photo/" +str(line) +'-fb.png'+ "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/str_photo/" + str(line)+ "-fb.png" + "*" + str(first_line[0])+ "*" +str(value)+ '\n')
                            f.write("/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/seq_photo/" +str(line) +'-f-l-r.png'+ "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/str_photo/" + str(line)+ "-f-l-r.png" + "*" + str(first_line[0])+ "*" +str(value)+ '\n')
                            f.write("/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/seq_photo/" +str(line) +'-fb-l-r.png'+ "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/str_photo/" + str(line)+ "-fb-l-r.png" + "*" + str(first_line[0])+ "*" +str(value)+ '\n')
                            f.write("/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/seq_photo/" +str(line) +'-f-u-d.png'+ "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/str_photo/" + str(line)+ "-f-u-d.png" + "*" + str(first_line[0])+ "*" +str(value)+'\n')
                            f.write("/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/seq_photo/" +str(line) +'-fb-u-d.png'+ "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/"+path+'/'+function+"/str_photo/" + str(line)+ "-fb-u-d.png" + "*" + str(first_line[0])+ "*" +str(value)+'\n')
                        else:
                            f.write(
                                "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/seq_photo/" + str(line) + '-f.png' + "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/str_photo/" + str(line) + "-f.png" + "*" + str(first_line[0]) + "*" + str(0) + '\n')
                            f.write(
                                "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/seq_photo/" + str(
                                    line) + '-fb.png' + "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/str_photo/" + str(
                                    line) + "-fb.png" + "*" + str(first_line[0]) + "*" + str(0) + '\n')
                            f.write(
                                "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/seq_photo/" + str(
                                    line) + '-f-l-r.png' + "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/str_photo/" + str(
                                    line) + "-f-l-r.png" + "*" + str(first_line[0]) + "*" + str(0) + '\n')
                            f.write(
                                "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/seq_photo/" + str(
                                    line) + '-fb-l-r.png' + "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/str_photo/" + str(
                                    line) + "-fb-l-r.png" + "*" + str(first_line[0]) + "*" + str(0) + '\n')
                            f.write(
                                "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/seq_photo/" + str(
                                    line) + '-f-u-d.png' + "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/str_photo/" + str(
                                    line) + "-f-u-d.png" + "*" + str(first_line[0]) + "*" + str(0) + '\n')
                            f.write(
                                "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/seq_photo/" + str(
                                    line) + '-fb-u-d.png' + "*" + "/home/chenjingjing/Models/MSFF-CDCGAN/picture/" + path + '/' + function + "/str_photo/" + str(
                                    line) + "-fb-u-d.png" + "*" + str(first_line[0]) + "*" + str(0) + '\n')

                            


name = 'bprna_new'
function='test'
# path='/home/chenjingjing/DATA/data/archiveII/shortdata/bpseq/'
# path='/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/train_set/'
path='/home/chenjingjing/DATA/data/bprna-new/without_pes/short/bpseq/'
datasets = Read_File(name,path,function)
datasets = Read_File2(name,path,function)