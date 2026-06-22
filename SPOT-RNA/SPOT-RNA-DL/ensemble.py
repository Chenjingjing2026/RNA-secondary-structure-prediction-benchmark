import argparse
import json
import os

import numpy as np
import torch.nn
from Bio import SeqIO

from constraints import prob_to_secondary_structure, one_hot
from handle_data import one_hot_encode, seqmatrix_coding
from loader import SPOTDataSetDot
from model import InputConvUnit, ResBlock, BiLSTM, BooleanMask, FCL
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
base_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(add_help=True)

parser.add_argument('--seq', dest='seq', type=str, nargs='?', required=True,
                    help='seq file(.fasta)')
parser.add_argument('--outputs', dest='outputs', type=str, nargs='?', required=True,
                    help='outputs')
args = parser.parse_args()
rna = SeqIO.parse(args.seq, 'fasta')
NUM_MODELS = [0, 1, 2, 4]
device = torch.device('cuda')
outputs = {}
mask = {}
sequences = {}

for seq in rna:
    sequences[seq.name] = seq.seq.replace(" ", "").upper().replace("T", "U")

for MODEL in NUM_MODELS:
    with open('model' + str(MODEL) + '.json', 'r') as f:
        config = json.load(f)
    seq_modules_list = [InputConvUnit(out_dim=config['chanNum'])]
    for i in range(config['BlockANum']):
        seq_modules_list.append(ResBlock(dim=config['chanNum'], dilation=config['dilaFactor']))

    if not config['disableSeq2seq']:
        seq_modules_list.append(BiLSTM(in_dim=config['chanNum'], hidden_dim=config['Seq2seqHiddenDim'], dropout=0))
        seq_modules_list.append(BooleanMask(dim=2 * config['Seq2seqHiddenDim']))
        seq_modules_list.append(FCL(in_dim=2 * config['Seq2seqHiddenDim'], hidden_layers=config['BlockBHiddenLayer'],
                                    hidden_dim=config['BlockBHiddenDim']))
    else:
        seq_modules_list.append(BooleanMask(dim=config['chanNum']))
        seq_modules_list.append(FCL(in_dim=config['chanNum'], hidden_layers=config['BlockBHiddenLayer'],
                                    hidden_dim=config['BlockBHiddenDim']))

    model = torch.nn.Sequential(*seq_modules_list).to(device=device)

    state_dict = torch.load('model' + str(MODEL) + '.model')
    model.load_state_dict(state_dict)
    model.eval()
    for name, sequence in sequences.items():
        seq = one_hot(sequence)
        seq = torch.asarray(seq, device=device, dtype=torch.float)
        seq = seqmatrix_coding(seq)
        pred = model(seq)

        mask[name] = torch.triu(torch.ones(len(sequence), len(sequence)), diagonal=2).detach().cpu().numpy()
        if MODEL == 0:
            outputs[name] = [torch.sigmoid(pred).detach().cpu().numpy()]
        else:
            outputs[name].append(torch.sigmoid(pred).detach().cpu().numpy())

RNA_ids = [i for i in list(outputs.keys())]
ensemble_outputs = {}

for i in RNA_ids:
    ensemble_outputs[i] = np.mean(outputs[i], 0)
    prob_to_secondary_structure(ensemble_outputs[i], mask[i], sequences[i], i, args, base_path)
