import numpy as np
import os.path as osp
import _pickle as cPickle
from tqdm import tqdm
from torch.utils import data
import pickle

def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)
        return attr


class RNADataset(data.Dataset):
    def __init__(self, path, dataname):
        self.path = path
        self.dataname = dataname
        self.data = self.cache_data

    def __len__(self):
        return len(self.data)

    def get_data(self, dataname):
        filename = dataname + '.pickle'
        # pre_data = cPickle.load(open(osp.join(self.path, filename), 'rb'))

        # 修改 cPickle 相关的代码

        pre_data = pickle.load(open(osp.join(self.path, filename), 'rb'))

        data = []
        for instance in tqdm(pre_data):
            data_x, _, seq_length, name, pairs = instance
            l = get_cut_len(seq_length, 80)
            # contact
            contact = np.zeros((l, l))
            contact[tuple(np.transpose(pairs))] = 1. if pairs != [] else 0.
            # data_seq
            data_seq = np.zeros((l, 4))
            data_seq[:seq_length] = data_x[:seq_length]
            data.append([contact, seq_length, data_seq,name])
        return data

    @cached_property
    def cache_data(self):
        return self.get_data(self.dataname)

    def __getitem__(self, index):
        return self.data[index]


# import os
# import numpy as np
# from pathlib import Path
# from collections import defaultdict
#
#
# class BpseqRNADataset(RNADataset):
#     def __init__(self, path, dataname, max_len=500):
#         self.bpseq_dir = Path(path) / 'bpseq'  # bpseq文件存放目录
#         self.max_len = max_len
#         super().__init__(path, dataname)
#
#     def parse_bpseq(self, filepath):
#         """解析单个bpseq文件"""
#         sequence = []
#         pairs = []
#         with open(filepath, 'r') as f:
#             for line in f:
#                 if line.startswith('#'):
#                     continue
#                 parts = line.strip().split()
#                 if len(parts) != 3:
#                     continue
#                 pos, base, pair = int(parts[0]), parts[1].upper(), int(parts[2])
#
#                 # 存储序列
#                 sequence.append(base)
#
#                 # 处理配对信息
#                 if pair != 0 and pos < pair:  # 避免重复记录
#                     pairs.append((pos - 1, pair - 1))  # 转换为0-based索引
#
#         return sequence, pairs
#
#     def get_data(self, dataname):
#         """重写数据加载方法"""
#         data = []
#         bpseq_files = list(self.bpseq_dir.glob('*.bpseq'))
#
#         for bp_file in tqdm(bpseq_files, desc='Processing bpseq'):
#             # 解析原始数据
#             seq, pairs = self.parse_bpseq(bp_file)
#             seq_length = len(seq)
#
#             # 长度过滤
#             if seq_length > self.max_len:
#                 continue
#
#             # 生成one-hot编码
#             base_dict = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
#             data_x = np.zeros((seq_length, 4))
#             for i, base in enumerate(seq):
#                 if base in base_dict:
#                     data_x[i, base_dict[base]] = 1
#                 else:  # 处理非常见碱基
#                     data_x[i, 0] = 1  # 默认设为A
#
#             # 动态填充长度
#             padded_len = get_cut_len(seq_length, 80)
#
#             # 填充序列
#             padded_seq = np.zeros((padded_len, 4))
#             padded_seq[:seq_length] = data_x
#
#             # 生成contact矩阵
#             contact = np.zeros((padded_len, padded_len))
#             if pairs:
#                 pairs = np.array(pairs)
#                 valid_pairs = pairs[(pairs[:, 0] < padded_len) & (pairs[:, 1] < padded_len)]
#                 contact[valid_pairs[:, 0], valid_pairs[:, 1]] = 1
#                 contact[valid_pairs[:, 1], valid_pairs[:, 0]] = 1  # 对称矩阵
#
#             data.append([
#                 contact.astype(np.float32),
#                 seq_length,
#                 padded_seq.astype(np.float32),
#                 bp_file.stem,  # 保留文件名作为标识
#                 np.array(pairs) if pairs else np.empty((0, 2))  # 保持与原代码兼容
#             ])
#
#         return data

# from .dataset import RNADataset
# from torch.utils.data import DataLoader
# from pathlib import Path
# import numpy as np
# import _pickle as cPickle
# from tqdm import tqdm
#
#
# class BpseqDataset(RNADataset):
#     """专门处理bpseq格式的数据集"""
#
#     def __init__(self, path, dataname, max_len=600):
#         # 你的实际bpseq路径
#         self.bpseq_dir = Path("/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/without_pseudoknot/short/test")
#         self.max_len = max_len
#         super().__init__(path, dataname)
#
#     def get_data(self, dataname):
#         data = []
#         # 扫描所有bpseq文件
#         bpseq_files = list(self.bpseq_dir.glob("*.bpseq"))
#
#         for bp_file in tqdm(bpseq_files, desc='Processing bpseq'):
#             # 解析bpseq文件
#             seq, pairs = self.parse_bpseq(bp_file)
#             seq_length = len(seq)
#
#             # 长度过滤
#             if seq_length > self.max_len or seq_length < 10:
#                 continue
#
#             # 生成one-hot编码
#             data_x = np.eye(4)[[self.base_to_index(b) for b in seq]]
#
#             # 动态填充长度
#             padded_len = ((seq_length - 1) // 16 + 1) * 16 if seq_length > 80 else 80
#
#             # 填充序列
#             padded_seq = np.zeros((padded_len, 4))
#             padded_seq[:seq_length] = data_x
#
#             # 生成contact矩阵
#             contact = np.zeros((padded_len, padded_len))
#             if pairs:
#                 valid_pairs = [(i, j) for i, j in pairs if i < padded_len and j < padded_len]
#                 contact[tuple(np.array(valid_pairs).T)] = 1
#                 contact += contact.T  # 保持对称性
#
#             data.append([
#                 contact.astype(np.float32),
#                 seq_length,
#                 padded_seq.astype(np.float32),
#                 bp_file.stem,
#                 np.array(pairs) if pairs else np.empty((0, 2))
#             ])
#         return data
#
#     @staticmethod
#     def base_to_index(base):
#         return {'A': 0, 'U': 1, 'C': 2, 'G': 3}.get(base, 0)
#
#     def parse_bpseq(self, filepath):
#         """解析单个bpseq文件"""
#         sequence = []
#         pairs = []
#         with open(filepath) as f:
#             for line in f:
#                 if line.startswith('#'):
#                     continue
#                 parts = line.strip().split()
#                 if len(parts) != 3:
#                     continue
#                 pos, base, pair = int(parts[0]), parts[1].upper(), int(parts[2])
#                 sequence.append(base)
#                 if pair != 0 and pos < pair:  # 避免重复记录
#                     pairs.append((pos - 1, pair - 1))  # 转换为0-based
#         return sequence, pairs
#
#
# def load_data(data_name, batch_size, data_root, num_workers=8, **kwargs):
#     # 原始数据加载逻辑
#     if data_name == 'RNAStralign':
#         test_set = RNADataset(path=data_root, dataname='train_512')
#     elif data_name == 'ArchiveII':
#         # 使用修改后的BpseqDataset
#         test_set = BpseqDataset(path=data_root, dataname='archiveII_bpseq')
#     elif data_name == 'bpRNA':
#         test_set = RNADataset(path=data_root, dataname='test')
#
#     # 保持原有DataLoader参数
#     loader = DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         drop_last=True,
#         collate_fn=lambda x: x  # 保持原始数据格式
#     )
#     return loader
#
# #
# # def load_data(data_name, batch_size, data_root, num_workers=8, **kwargs):
# #     # 原始数据加载逻辑
# #     if data_name == 'RNAStralign':
# #         test_set = RNADataset(path=data_root, dataname='test_600')
# #     elif data_name == 'ArchiveII':
# #         # 使用修改后的BpseqDataset
# #         test_set = BpseqDataset(path=data_root, dataname='archiveII_bpseq')
# #     elif data_name == 'bpRNA':
# #         test_set = RNADataset(path=data_root, dataname='test')
# #
# #     # 保持原有DataLoader参数
# #     loader = DataLoader(
# #         test_set,
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=num_workers,
# #         drop_last=True,
# #         collate_fn=lambda x: x  # 保持原始数据格式
# #     )
# #     return loader
#
# # # 使用示例
# # if __name__ == '__main__':
# #     # 初始化数据集
# #     dataset = BpseqRNADataset(
# #         path='/home/chenjingjing/DATA/data/archiveII/shortdata/bpseq',
# #         dataname='custom_bpseq',
# #         max_len=600  # 根据需求调整最大长度
# #     )
# #
# #     # 验证数据样本
# #     sample = dataset[0]
# #     print(f"Sample Info:")
# #     print(f"Contact shape: {sample[0].shape}")
# #     print(f"Seq length: {sample[1]}")
# #     print(f"Padded seq shape: {sample[2].shape}")
# #     print(f"Filename: {sample[3]}")
# #     print(f"Pairs: {sample[4][:5]}... (total {len(sample[4])} pairs)")
#
# # # 初始化加载器
# # loader = load_data(
# #     data_name='RNAStralign',  # 触发BpseqDataset
# #     batch_size=1,
# #     data_root='/home/chenjingjing/DATA/data/archiveII/shortdata',
# #     num_workers=4
# # )
# #
# #
# # # 原始数据加载逻辑
# # if data_name == 'RNAStralign':
# #     test_set = RNADataset(path=data_root, dataname='test_600')
# # elif data_name == 'ArchiveII':
# #     # 使用修改后的BpseqDataset
# #     test_set = BpseqDataset(path=data_root, dataname='archiveII_bpseq')
# # elif data_name == 'bpRNA':
# #     test_set = RNADataset(path=data_root, dataname='test')
# # # 验证批次数据
# # for batch in loader:
# #     contact, lengths, seqs, names, pairs = batch[0]  # batch_size=1时
# #     print(f"Batch info: {names[0]}, Length: {lengths[0]}")
# #     print(f"Contact map shape: {contact.shape}")
# #     break
#
# # 正确触发BpseqDataset的验证代码
# if __name__ == '__main__':
#     # 初始化加载器
#     loader = load_data(
#         data_name='bpRNA',  # ✅ 正确触发条件
#         batch_size=1,
#         data_root='/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/without_pseudoknot/short/test/',
#         num_workers=4
#     )
#
#     # 验证第一个批次
#     for batch in loader:
#         contact, length, seq, name, pairs = batch[0]  # batch_size=1时直接取[0]
#         print(f"Name: {name} | Length: {length}")
#         print(f"Contact map shape: {contact.shape}")
#         print(f"Example pairs: {pairs[:5]}")
#         break