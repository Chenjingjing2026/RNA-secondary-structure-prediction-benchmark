# -*- coding: utf-8 -*-
import collections
import os
import pickle as cPickle
from os.path import join
from random import shuffle
from torch.utils import data
from itertools import product
from typing import List, Tuple

import sys
sys.path.append('/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/common')
from data_utils import *
import bisect

perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1, 3], [3, 1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]


def make_dataset(
        directory: str
) -> List[str]:
    instances = []
    directory = os.path.expanduser(directory)
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):
            if fname.endswith('.cPickle') or fname.endswith('.Pickle'):
                path = os.path.join(root, fname)
                instances.append(path)

    return instances



def get_data_fcn(data_seq, data_length, set_length):
    perm = list(product(np.arange(4), np.arange(4)))
    data_fcn = np.zeros((16, set_length, set_length))
    for n, cord in enumerate(perm):
        i, j = cord
        data_fcn[n, :data_length, :data_length] = np.matmul(
            data_seq[:data_length, i].reshape(-1, 1),
            data_seq[:data_length, j].reshape(1, -1)
        )
    data_fcn_1 = np.zeros((1, set_length, set_length))
    data_fcn_1[0, :data_length, :data_length] = creatmat(data_seq[:data_length, :])
    data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)

    return data_fcn_2

class ParserData(object):
    def __init__(self, path):
        self.path = path
        self.data = self.load_data(self.path)
        self.len = len(self.data)
        self.seq_max_len = max([len(x.seq_raw) for x in self.data])
        self.set_max_len = (self.seq_max_len // 80 + int(self.seq_max_len % 80 != 0)) * 80  # 80的倍数

    def load_data(self, path):
        # RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')
        with open(path, 'rb') as f:
            load_data = cPickle.load(f)
        return load_data

    def padding(self, data_array, maxlen):
        a, b = data_array.shape
        return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')

    def pairs2map(self, pairs, seq_len):
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def get_item(self,idx):
        # shuffle(self.data)
        # 以下被我修改过。。。
        # data_list=[]   
        # for item in self.data:   # TODO： 把这些东西都转成相应的tensor，方便进行转换。
        item=self.data[idx]            
        data_dict={}
        # get_data_fcn()
        # data_dict['data_fcn_2']=torch.tensor(get_data_fcn(seq_encoding(item['seq_str']), len(item['seq_str']), self.set_max_len)).float()
        data_dict['data_fcn_2']=seq_encoding(item.seq_raw)
        # data_dict['data_fcn_2']=torch.tensor(item[0]).float()
        data_dict['data_seq_raw']=item.seq_raw  # 似乎不用变
        data_dict['data_length']=torch.tensor(len(item.seq_raw)).long()
        data_dict['data_name']=item.name
        data_dict['contact']=torch.tensor(pairs2map(item.contact, self.set_max_len)).long()   #为啥要unsqueeze(1) ?
        data_dict['data_seq_encode_pad']=torch.tensor(padding(seq_encoding(item.seq_raw), self.set_max_len))  # ????
        data_dict['set_max_len']=self.set_max_len  # 不要用
        # data_list.append(data_dict)
        # contact_list = [item[4] for item in self.data]
        # data_fcn_2_list = [item[0] for item in self.data]
        # data_seq_raw_list = [item[1] for item in self.data]
        # data_length_list = [item[2] for item in self.data]
        # data_name_list = [item[3] for item in self.data]
        # contact_list=[pairs2map(_list, self.set_max_len) for _list in contact_list]  # fix
        # contact_array = np.stack(contact_list, axis=0)
        # data_fcn_2_array = np.stack(data_fcn_2_list, axis=0)

        # data_seq_encode_list = list(map(lambda x: seq_encoding(x), data_seq_raw_list))
        # data_seq_encode_pad_list = list(map(lambda x: self.padding(x, self.set_max_len), data_seq_encode_list))
        # data_seq_encode_pad_array = np.stack(data_seq_encode_pad_list, axis=0)

        return data_dict

# 因为数据集相对比较少，所以直接全部都读取进来
class Dataset(data.Dataset):

    def __init__(
            self,
            data_file_paths: List[str],
            upsampling: bool = False
    ) -> None:
        self.data_file_paths=data_file_paths
        self.upsampling = upsampling

        samples = []
        for data_file in self.data_file_paths:
            samples.append(data_file)


        self.file_samples = samples  # 这个samples是一个列表，里面存放的是数据的路径
        # if self.upsampling:
        #     self.file_samples = self.upsampling_data_files()  # 增加更长的数据的比例，
        
        
        self.data=[]
        self.offset_list=[0]
        
                   
        for sample_file in self.file_samples:
            file_data = ParserData(sample_file)
            self.offset_list.append(file_data.len+self.offset_list[-1])   
            self.data.append(file_data) 
            

            # data_list = file_data.preprocess_data()
            

            # contact = torch.tensor(contact_array).unsqueeze(1).long()
            # data_fcn_2 = torch.tensor(data_fcn_2_array).float()
            # data_length = torch.tensor(data_length_list).long()
            # data_seq_encode_pad = torch.tensor(data_seq_encode_pad_array).float()
            # self.data.extend(data_list)

        self.__length= self.offset_list[-1]

    @staticmethod
    def make_dataset(
            directory: str
    ) -> List[str]:
        return make_dataset(directory)

    # for data balance, 4 times for 160~320 & 320~640
    def upsampling_data_files(self):
        # RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 'data_fcn_2 seq_raw name length contact')  #Fix 顺序  失败了QAQ  这玩意顺序对不上，mei
        augment_data_list = list()
        final_data_list = self.file_samples
        for data_path in final_data_list:
            with open(data_path, 'rb') as f:
                load_data = cPickle.load(f)
            max_len = max([x[2] for x in load_data])
            if max_len <= 160:
                continue
            else:
                augment_data_list.append(data_path)
            # elif max_len == 320:
                # augment_data_list.append(data_path)
            # elif max_len == 640:
                # augment_data_list.append(data_path)

        augment_data_list = list(np.random.choice(augment_data_list, 3 * len(augment_data_list))) # 把要增强的数据集重复3倍
        final_data_list.extend(augment_data_list)
        shuffle(final_data_list)
        return final_data_list

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return self.__length

    def __getitem__(self, index: int):
        # output_dict=self.data[index]
        
        # batch_data = ParserData(batch_data_path)

        # contact_array, data_fcn_2_array, data_seq_raw_list, data_length_list, data_name_list, set_max_len, \
        # data_seq_encode_pad_array = batch_data.preprocess_data()

        # contact = torch.tensor(contact_array).unsqueeze(1).long()
        # data_fcn_2 = torch.tensor(data_fcn_2_array).float()
        # data_length = torch.tensor(data_length_list).long()
        # data_seq_encode_pad = torch.tensor(data_seq_encode_pad_array).float()

        # return contact, data_fcn_2, data_seq_raw_list, data_length, data_name_list, set_max_len, data_seq_encode_pad
        # 根据offset_list找到对应的文件
        dset_idx = bisect.bisect_left(self.offset_list, index)-1  # 
        
        item=self.data[dset_idx].get_item(index-self.offset_list[dset_idx])
        
        return item


def generate_token_batch(alphabet, seq_strs):  # 转tokens,便于输入RNA-FM模型
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len
            + int(alphabet.prepend_bos)
            + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    for i, seq_str in enumerate(seq_strs):
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[i, int(alphabet.prepend_bos): len(seq_str) + int(alphabet.prepend_bos), ] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_str) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens

'''

这个函数 diff_collate_fn 是一个自定义的 ​批次处理函数（collate function）​，主要用于处理不同长度的序列数据，将它们填充（padding）到统一长度并拼接成批次张量。以下是其核心功能的逐步说明：
'''
def diff_collate_fn(batch, alphabet):  # 直接填充？不要分数据集？  这个应该被修改 TODO: 写信的coolate_fn按照这个batch内的最大长度来进行填充
    # contact, data_fcn_2, data_seq_raw_list, data_length, data_name_list, set_max_len, data_seq_encode_pad = zip(*batch)
    '''
    data_fcn_2和 data_seq_encode_pad 重复了？
    '''
    set_max_len = max(item['set_max_len'] for item in batch)
    # set_max_len = max([len(item['data_seq_raw']) for item in batch])  # 获取最大的长度？
    # set_max_len = max([len(item['data_seq_raw']) for item in batch])  # 遵循源码意愿
    set_max_len = (set_max_len // 80 + int(set_max_len % 80 != 0)) * 80  # 80的倍数
    collated_batch={}
    seq_strs = [item['data_seq_raw'] for item in batch]
    
    for item in batch:   # 先把所有的数据都合并到一个字典里
        for key in item.keys():
            if key not in collated_batch:
                collated_batch[key]=[]
            target=item[key]
            if(key=='contact'):
                target=F.pad(target,(0,set_max_len-target.shape[-2],0,set_max_len-target.shape[-1]),'constant',0)
            elif(key=='data_fcn_2'):
                target=get_data_fcn(target, item['data_length'], set_max_len)
                target=torch.tensor(target).float()
                # target=F.pad(target,(0,0,0,set_max_len-target.shape[-2]),'constant',0)  # 对倒数第二个维度L进行填充
            elif(key=='data_seq_encode_pad'):
                target=F.pad(target,(0,0,0,set_max_len-target.shape[-2]),'constant',0)
            
            # elif(key == 'data_seq_raw'):
                # collated_batch[key].append(target)
                # key='tokens'  # 转变成tokens
                # if key not in collated_batch:
                    # collated_batch[key]=[]
                # target = generate_token_batch(alphabet, target)
                # target = F.pad(target, (0, set_max_len - target.shape[-1]), 'constant', 0)
                
            collated_batch[key].append(target) 
               
    collated_batch['tokens'] = generate_token_batch(alphabet, seq_strs)
    collated_batch['contact']=torch.stack(collated_batch['contact'],dim=0)
    collated_batch['data_fcn_2']=torch.stack(collated_batch['data_fcn_2'],dim=0)
    # torch.stack(collated_batch['tokens'],dim=0)
    collated_batch['data_length']=torch.stack(collated_batch['data_length'],dim=0)
    collated_batch['data_seq_encode_pad']=torch.stack(collated_batch['data_seq_encode_pad'])
    collated_batch['set_max_len']=set_max_len
    return collated_batch
        
        
    # if len(contact) == 1:
    #     contact = contact[0]
    #     data_fcn_2 = data_fcn_2[0]
    #     data_seq_raw = data_seq_raw_list[0]
    #     data_length = data_length[0]
    #     data_name = data_name_list[0]
    #     set_max_len = set_max_len[0]
    #     data_seq_encode_pad = data_seq_encode_pad[0]

    # else:
    #     set_max_len = max(set_max_len) if isinstance(set_max_len, tuple) else set_max_len

    #     contact_list = list()
    #     for item in contact:
    #         if item.shape[-1] < set_max_len:
    #             item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
    #             contact_list.append(item)
    #         else:
    #             contact_list.append(item)

    #     data_fcn_2_list = list()
    #     for item in data_fcn_2:
    #         if item.shape[-1] < set_max_len:
    #             item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
    #             data_fcn_2_list.append(item)
    #         else:
    #             data_fcn_2_list.append(item)

    #     data_seq_encode_pad_list = list()
    #     for item in data_seq_encode_pad:
    #         if item.shape[-1] < set_max_len:
    #             item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
    #             data_seq_encode_pad_list.append(item)
    #         else:
    #             data_seq_encode_pad_list.append(item)

        # contact = torch.cat(contact_list, dim=0)
        # data_fcn_2 = torch.cat(data_fcn_2_list, dim=0)
        # data_seq_encode_pad = torch.cat(data_seq_encode_pad_list, dim=0)

        # data_seq_raw = list()
        # for item in data_seq_raw_list:
        #     data_seq_raw.extend(item)

        # data_length = torch.cat(data_length, dim=0)

        # data_name = list()
        # for item in data_name_list:
        #     data_name.extend(item)


    return contact, data_fcn_2, tokens, data_length, data_name, set_max_len, data_seq_encode_pad



# def diff_collate_fn(batch, alphabet):  # 直接填充？不要分数据集？  这个应该被修改
#     contact, data_fcn_2, data_seq_raw_list, data_length, data_name_list, set_max_len, data_seq_encode_pad = zip(*batch)
#     if len(contact) == 1:
#         contact = contact[0]
#         data_fcn_2 = data_fcn_2[0]
#         data_seq_raw = data_seq_raw_list[0]
#         data_length = data_length[0]
#         data_name = data_name_list[0]
#         set_max_len = set_max_len[0]
#         data_seq_encode_pad = data_seq_encode_pad[0]

#     else:
#         set_max_len = max(set_max_len) if isinstance(set_max_len, tuple) else set_max_len

#         contact_list = list()
#         for item in contact:
#             if item.shape[-1] < set_max_len:
#                 item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
#                 contact_list.append(item)
#             else:
#                 contact_list.append(item)

#         data_fcn_2_list = list()
#         for item in data_fcn_2:
#             if item.shape[-1] < set_max_len:
#                 item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
#                 data_fcn_2_list.append(item)
#             else:
#                 data_fcn_2_list.append(item)

#         data_seq_encode_pad_list = list()
#         for item in data_seq_encode_pad:
#             if item.shape[-1] < set_max_len:
#                 item = F.pad(item, (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]), 'constant', 0)
#                 data_seq_encode_pad_list.append(item)
#             else:
#                 data_seq_encode_pad_list.append(item)

#         contact = torch.cat(contact_list, dim=0)
#         data_fcn_2 = torch.cat(data_fcn_2_list, dim=0)
#         data_seq_encode_pad = torch.cat(data_seq_encode_pad_list, dim=0)

#         data_seq_raw = list()
#         for item in data_seq_raw_list:
#             data_seq_raw.extend(item)

#         data_length = torch.cat(data_length, dim=0)

#         data_name = list()
#         for item in data_name_list:
#             data_name.extend(item)

#     tokens = generate_token_batch(alphabet, data_seq_raw)

#     return contact, data_fcn_2, tokens, data_length, data_name, set_max_len, data_seq_encode_pad


def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')


def pairs2map(pairs, seq_len):
    contact = np.zeros([seq_len, seq_len])
    for pair in pairs:
        contact[pair[0], pair[1]] = 1
    return contact


def Gaussian(x):
    return math.exp(-0.5 * (x * x))


def paired(x, y):
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0


def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat
