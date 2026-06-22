import pickle
import sys
sys.path.append('/home/chenjingjing/Models/GCNfold/GCNfold-master/data')
# from RNAGraph import DGLFormDataset
# from RNAGraph import RNADataset

# 指定文件路径
file_path = '/home/chenjingjing/Models/GCNfold/GCNfold-master/data/archiveII_all/archiveII_all.pkl'

# 加载pkl文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 查看数据的内容
print(data)
