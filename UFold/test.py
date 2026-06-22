import pickle
import collections
import numpy as np
RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')

with open(r'/home/chenjingjing/Models/UFold/UFold/data/TS2.cPickle','rb') as file:
    data = pickle.load(file,encoding='iso-8859-1')
    print('样本数:', len(data))
    data_x = np.array([instance[0] for instance in data])
    data_y = np.array([instance[1] for instance in data])
    for instance in data:
        test = instance[4]
        print(len(test))
    pairs = np.array([instance[4] for instance in data])
    print(data_x.shape)
    print(data_y.shape)
    print(pairs.shape)

with open(r'/home/chenjingjing/Models/UFold/UFold/data/ArchiveII/ArchiveII.cPickle','rb') as file:
    data = pickle.load(file)
    print('样本数:', len(data))
    data_x = np.array([instance[0] for instance in data])
    data_y = np.array([instance[1] for instance in data])
    pairs = np.array([instance[-1] for instance in data])
    print(data_x.shape)
    print(data_y.shape)
    print(pairs.shape)

# import cPickle as pickle; data=pickle.load(open('data/your_datasprint('样本数:', len(data)); print('第一个样本形状:', [x.shape for x in data[0]])et_name.cPickle','rb')); "