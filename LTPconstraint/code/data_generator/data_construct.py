import numpy as np
import pandas as pd
import os
File_dir = "/home/chenjingjing/DATA/data/RNAStrAlign/shortdata/bpseq/"
Save_dir = "/home/chenjingjing/Models/LTPConstraint/data/128/"
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
word_coder = {
    "A":1,
    "U":2,
    "G":3,
    "C":4,
}
def generate_dataset(original_list, limit_length):
    seq = []
    label = []
    for i in range(limit_length):
        temp = []
        for j in range(limit_length):
            temp.append(0)
        label.append(temp)
    for ele in original_list:
        seq.append(word_coder.get(ele[1], 0))
        if(ele[4]!=0):
            x = ele[0]-1
            y = ele[4]-1
            label[x][y]=1
            label[y][x]=1
    for i in range(len(original_list), limit_length):
        seq.append(0)
    return seq, label
seq_list = []
label_atrix = []
limit_length = 128
kk=0
for f_name in os.listdir(File_dir):
    print('now processing:'+str(kk))
    kk = kk+1
    ele = pd.read_csv(File_dir+f_name,sep="\s+",skiprows=1, header=None)
    ele = ele.values.tolist()
    if len(ele) <= 128:
        _seq, _label = generate_dataset(ele, limit_length)
        seq_list.append(_seq)
        label_atrix.append(_label)
print('now traslating')
seq_list = np.array(seq_list)
label_atrix = np.array(label_atrix, dtype=np.float32)
print(f"Seq List Length: {len(seq_list)}")
print(f"Label Matrix Length: {len(label_atrix)}")

print(seq_list.shape, label_atrix.shape)

from numpy import savez, loadtxt
np.savez(Save_dir+'RNAStrAlign', seq_list, label_atrix)