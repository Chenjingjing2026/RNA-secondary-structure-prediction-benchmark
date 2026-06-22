import pickle
import os
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

pickle_file_path = '/home/chenjingjing/Models/UFold/UFold/data/crossfamily/except_16s/train.pickle'  # 替换为实际文件路径
try:
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
        print(data)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading pickle file: {e}")
