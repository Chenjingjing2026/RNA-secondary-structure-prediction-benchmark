# -*- coding:utf-8 -*-

import os

import torch
import torch as t
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import numpy as np

np.random.seed(1080)


class SPOTDataSetDot(data.Dataset):
    def __init__(self, dataset_folder, device,flist):
        'Initialization'
        super(SPOTDataSetDot, self).__init__()
        self.dataset_folder = dataset_folder
        self.device = device
        self.flist = flist
        self.samples = self.load_sample_list()

    def load_sample_list(self):
        samples = []
        for line in open(os.path.join(self.dataset_folder, self.flist), 'r'):
            samples.append(line)
        return samples

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def handle_dot_matrix(self, dot_matrix):
        n = len(dot_matrix)
        matrix = torch.zeros(size=(n, n))
        for i, struct in enumerate(dot_matrix):
            if struct != 0:
                j = struct - 1
                matrix[i, j] = 1
                matrix[j, i] = 1
        mask = t.triu(t.ones(n, n), diagonal=2)
        res = matrix[mask == 1].unsqueeze(-1)
        return res

    def seqmatrix_coding(self, seq_matrix):
        n = seq_matrix.shape[0]
        one_hot = seq_matrix.unsqueeze(0)
        one_hot = one_hot.repeat([n, 1, 2])
        one_hot[:, :, 4] = one_hot[:, :, 0].t()
        one_hot[:, :, 5] = one_hot[:, :, 1].t()
        one_hot[:, :, 6] = one_hot[:, :, 2].t()
        one_hot[:, :, 7] = one_hot[:, :, 3].t()
        one_hot = one_hot.permute([2, 0, 1])
        one_hot = one_hot.unsqueeze(0).contiguous()
        return one_hot

    def __getitem__(self, index):
        'Generates one sample of data'
        rna_name, seq_fn, dot_fn = self.samples[index].strip().split()
        seq_fn = os.path.join(self.dataset_folder, seq_fn)
        dot_fn = os.path.join(self.dataset_folder, dot_fn)
        seq_matrix = t.FloatTensor(np.fromfile(seq_fn, sep=',').reshape(-1, 4))
        dot_matrix = t.LongTensor(np.fromfile(dot_fn, sep=',').reshape(-1, 1))
        dot_matrix = self.handle_dot_matrix(dot_matrix)
        seq_matrix = seq_matrix.to(self.device)
        dot_matrix = dot_matrix.to(self.device)
        seq_matrix = self.seqmatrix_coding(seq_matrix)

        return (rna_name, seq_matrix), dot_matrix
