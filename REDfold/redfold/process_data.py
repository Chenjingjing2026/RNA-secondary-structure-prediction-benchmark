#! /usr/bin/python3
# Data Process from process_data_newdataset.py


import numpy as np
import subprocess
import collections
import pickle as cPickle
import random
import time
import sys
import os

from Bio import SeqIO
from itertools import product, combinations
from .fold import *
from .fold.config import process_config




def one_hot(seq1):
    RNA_seq= seq1

    feat= np.concatenate([[(npBASE1 == base.upper()).astype(int)] 
          if str(base).upper() in BASE1 else np.array([[0] * len(BASE1)]) for base in RNA_seq])

    return feat


# def one_hot_2m(seq1):
#     L1= len(seq1)
#     feat= np.zeros((L1,16))
#     for i in range(0,L1-1):
#       Id1= str(seq1[i:i+2]).upper()
#       if Id1 in dcBASE2:
#         feat[i,dcBASE2[Id1]]= 1
#     #Circle Back 2mer
#     Id1= str(seq1[-1]+seq1[0]).upper()
#     feat[L1-1,dcBASE2[Id1]]= 1
#
#     return feat

def one_hot_2m(seq1):
  L1 = len(seq1)
  feat = np.zeros((L1, 16))

  for i in range(0, L1 - 1):
    Id1 = str(seq1[i:i + 2]).upper()
    # 使用 get() 方法，避免 KeyError
    if Id1 in dcBASE2:
      feat[i, dcBASE2[Id1]] = 1
    else:
      print(f"Warning: '{Id1}' not found in dcBASE2")  # 可选的警告信息
      continue

  # Circle Back 2mer
  Id1 = str(seq1[-1] + seq1[0]).upper()
  # 使用 get() 方法，避免 KeyError
  if Id1 in dcBASE2:
    feat[L1 - 1, dcBASE2[Id1]] = 1
  else:
    print(f"Warning: '{Id1}' not found in dcBASE2")  # 可选的警告信息
  

  return feat


def get_cut_len(data_len,set_len):
    L= data_len
    if L<= set_len:
        L= set_len
    else:
        L= (((L - 1) // 16) + 1) * 16
    return L


#- Check standard pairs
def check_stand(pairs, seq):
  for pair in pairs:
    str1= seq[pair[0]]+seq[pair[1]]
    if (str1 not in pair_set):
      print(f"Error: Pair({pair})->{str1}")
      continue
      # return False
      
  return True


def pair2map(pairs, seq_len):

  pmap= np.zeros([seq_len, seq_len])
  for pair in pairs:
    pmap[pair[0], pair[1]] = 1
  return pmap




#-process_data
def process_data(args,file_dir,file_conf):

  #-Param
  global npBASE1
  global dcBASE2


#  if not os.path.exists(file_conf):  
  config = process_config(file_conf)
  test_data= config.test_data
  out_step= config.out_step
#  file_dir= file1

  file2= config.data_type  
#  all_files= sorted(os.listdir(file_dir))
  all_files= os.listdir(file_dir)


  if not os.path.exists(file2):
    os.makedirs(file2)


  # RNA data Container
  all_files_list = []

  npBASE1= np.array([b1 for b1 in BASE1])
  npBASE2= np.array(["".join(b2) for b2 in product(npBASE1,npBASE1)])
  dcBASE2= {};
  for [a,b] in enumerate(npBASE2):
    dcBASE2[b]= a

  ##Add fasta file
  #pdb_file = open('pdb_yxc_list_train_672.fa','w')
  ## end add fasta file


  for index,item_file in enumerate(all_files):
    # Extract Sequence
    if (args.train):
      t0= subprocess.getstatusoutput('awk \'{print $2}\' '+file_dir+item_file)
      t1= subprocess.getstatusoutput('awk \'{print $1}\' '+file_dir+item_file)
      t2= subprocess.getstatusoutput('awk \'{print $3}\' '+file_dir+item_file)
      seq= ''.join(t0[1].split('\n'))
    else:
      t0,t1,t2=[[0],[1],[1]]
      seq= SeqIO.read(file_dir+item_file, "fasta").seq

    seq=seq.upper()
    # Transfer sequence to one-hot matrix [AUCG]
    if t0[0] == 0:
      try:
        one_hot_matrix= one_hot(seq.upper())
        one_hot_mat2= one_hot_2m(seq.upper())
      except IndexError as ie:
        print(f"{ie}")


    # List [Basepair Dict]
    # if t1[0] == 0 and t2[0] == 0:
    #   pair_dict_all_list = [[int(item_tmp)-1,int(t2[1].split('\n')[index_tmp])-1] for index_tmp,item_tmp in enumerate(t1[1].split('\n')) if int(t2[1].split('\n')[index_tmp]) != 0]
    #   # print(f"seq_name:{seq_name}")
    #   # print(f"seq_len:{seq_len}")
    # else:
    #   pair_dict_all_list = []

    # List [Basepair Dict]
    if t1[0] == 0 and t2[0] == 0:
      lines1 = t1[1].split('\n')
      lines2 = t2[1].split('\n')

      # 动态检测是否需要跳过表头（如果第一行不是数字，则跳过）
      skip_header = False
      if (len(lines1) > 0 and not lines1[0].strip().isdigit()) or \
              (len(lines2) > 0 and not lines2[0].strip().isdigit()):
        skip_header = True

      if skip_header:
        lines1 = lines1[1:] if len(lines1) > 1 else []
        lines2 = lines2[1:] if len(lines2) > 1 else []

      # 确保 lines1 和 lines2 长度一致
      min_len = min(len(lines1), len(lines2))
      lines1 = lines1[:min_len]
      lines2 = lines2[:min_len]

      pair_dict_all_list = [
        [int(item_tmp) - 1, int(lines2[index_tmp]) - 1]
        for index_tmp, item_tmp in enumerate(lines1)
        if (
                index_tmp < len(lines2)  # 防止越界
                and lines2[index_tmp].strip().isdigit()  # 确保是数字（并去除空格）
                and int(lines2[index_tmp]) != 0  # 过滤掉 0
        )
      ]
    else:
      pair_dict_all_list = []

    n_item= item_file.rfind('.')
    if n_item!=-1:
      seq_name= item_file[:n_item]
    else:
      seq_name = item_file
    seq_len = len(seq)
    print(f"seq_name:{seq_name}")
    # print(f"seq_len:{seq_len}")


    #Dict (basepairs)/remove half reverse list
    pair_dict_all = dict([item for item in pair_dict_all_list if item[0]<item[1]])

    if not (check_stand(pair_dict_all_list,seq)):
      exit()

    if index%out_step==0:
      print('current processing %d/%d'%(index+1,len(all_files)))

    if seq_len> 0 and seq_len<= 720:
      ss_label = np.zeros((seq_len,3),dtype=int)
      ss_label[[*pair_dict_all.keys()],] = [0,1,0]
      ss_label[[*pair_dict_all.values()],] = [0,0,1]

      L= get_cut_len(seq_len,80)

      ##-Trans seq to seq_length
      one_hot_matrix_LM= np.zeros((L,4))
      one_hot_matrix_LM[:seq_len,]= one_hot_matrix
#      ss_label_L= np.zeros((L,3),dtype=int)

      one_hot_mat2_LM= np.zeros((L,16))
      one_hot_mat2_LM[:seq_len,]= one_hot_mat2
      
      
      
      # default non-pair
#      ss_label_L[:]= [1,0,0]
#      ss_label_L[:seq_len,]= ss_label
#      ss_label_L[np.where(np.sum(ss_label_L,axis=1)<= 0)[0],]= [1,0,0]
      ##-End Trans sequnce
      data_seq1= one_hot_matrix_LM;
      data_seq2= one_hot_mat2_LM;

      
      ##-Seq_onehot
      seq_hot= one_hot_matrix_LM[:L,:]
      data_pair= pair2map(pair_dict_all_list,L)


      # RNA_SS_data
      sample_tmp= RNA_SS_data(name=seq_name, length=seq_len, seq_hot=seq_hot, data_pair=data_pair, data_seq1= data_seq1, data_seq2=data_seq2)      
      all_files_list.append(sample_tmp)

  print(f"All_file_list:{len(all_files_list)}")


  # Save RNA Data
  file2+= f"/{test_data}.pick"
  print(f"Save RNA test data to {file2}")
  random.shuffle(all_files_list)

  cPickle.dump(all_files_list,open(file2,"wb"))



if __name__=='__main__':

  try:
    args= get_args()
  except IndexError as ie:
    print(f"Warning: {ie}")





