import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data
from collections import defaultdict
import numpy as np

# from FCN import FCNNet
from Network import U_Net as FCNNet
# from Network3 import U_Net_FP as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset
# from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections

args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess


def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis=1).clamp_max(1))
    seq[contact.sum(axis=1) == 0] = -1
    return seq


def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file


def get_ct_dict(predict_matrix, batch_num, ct_dict):
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:, i, j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i, j)]
                else:
                    ct_dict[batch_num] = [(i, j)]
    return ct_dict


def get_ct_dict_fast(predict_matrix, batch_num, ct_dict, dot_file_dict, seq_embedding, seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1),
                        predict_matrix.cpu().sum(axis=1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1
    # seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp + 1).squeeze())
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    letter = 'AUCG'
    ct_dict[batch_num] = [(seq[0][i], seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    seq_letter = ''.join([letter[item] for item in np.nonzero(seq_embedding)[:, 1]])
    dot_file_dict[batch_num] = [(seq_name, seq_letter, dot_list[:len(seq_letter)])]
    return ct_dict, dot_file_dict


def evaluate_exact_new(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a) * torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp

    # total_samples = len(pred_a)  # 总样本数
    tn = (torch.Tensor(true_a) == 0).sum() - fp  # TN计算方法：真正为负的样本数

    # tn = total_samples - (tp + fp + fn)  # TN = 总样本数 - (TP + FP + FN)
    # recall = tp/(tp+fn)
    # precision = tp/(tp+fp)
    # f1_score = 2*tp/(2*tp + fp + fn)
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    # 计算 MCC
    mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
                tn + fp) * (tn + fn) > 0 else 0
    return precision, recall, f1_score, mcc

def load_base_pairs_info():
    # 创建目标位置集合
    target_positions = defaultdict(set)
    # for pair_idx in range(len(bp1_list)):
    #     pos1 = bp1_list[pair_idx] - 1  # 转换为0-based索引
    #     pos2 = bp2_list[pair_idx] - 1  # 转换为0-based索引
    #     target_positions.add((pos1, pos2))

    base_pairs_dict1 = defaultdict(set)
    base_pairs_dict2 = defaultdict(set)
    base_pairs_file= '/home/chenjingjing/DATA/bpRNA-1m/bpRNA_1m/Filtered/pseudoknot/short/all_pk_pairs.txt'

    try:
        with open(base_pairs_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        filename = parts[0]
                        pos1 = int(parts[2])
                        pos2 = int(parts[4])
                        target_positions[filename].add((pos1, pos2))
                        # base_pairs_dict1[filename].add(pos1)
                        # base_pairs_dict2[filename].add(pos2)
    except Exception as e:
        print(f"Warning: Error loading base pairs file: {e}")
    return dict(target_positions)
    # return dict(base_pairs_dict1),dict(base_pairs_dict2)
target_positions = load_base_pairs_info()

def evaluate_model_separate_bp(pred_a, true_a,name):
    target_correct = 0
    target_total = 0
    target_tp, target_fp, target_tn, target_fn = 0, 0, 0, 0

    # 初始化其他碱基对的指标
    other_total_loss = 0.0
    other_correct = 0
    other_total = 0
    other_tp, other_fp, other_tn, other_fn = 0, 0, 0, 0

    name = name[0].replace('.bpseq','')
    # pred_a = pred_a.cpu().reshape(-1)
    # true_a = true_a.cpu().reshape(-1)

    # 遍历每个位置
    for j in range(pred_a.shape[0]):
        for i in range(pred_a.shape[1]):
            pred = pred_a[j, i]
            true_val = true_a[j, i]

            if (j + 1, i + 1) in target_positions[name]:
                # 目标位置的混淆矩阵更新
                if pred == 1 and true_val == 1:
                    target_tp += 1
                elif pred == 1 and true_val == 0:
                    target_fp += 1
                elif pred == 0 and true_val == 0:
                    target_tn += 1
                elif pred == 0 and true_val == 1:
                    target_fn += 1
                print(f'{name} position ({j + 1},{i + 1}): tp{target_tp}, fp{target_fp}, tn{target_tn}, fn{target_fn}')
            else:
                # 非目标位置的混淆矩阵更新
                if pred == 1 and true_val == 1:
                    other_tp += 1
                elif pred == 1 and true_val == 0:
                    other_fp += 1
                elif pred == 0 and true_val == 0:
                    other_tn += 1
                elif pred == 0 and true_val == 1:
                    other_fn += 1

            # if j in base_pair1[name]:
            #     # 更新混淆矩阵
            #     if pred == 1 and true_val == 1:
            #         target_tp += 1
            #     elif pred == 1 and true_val == 0:
            #         target_fp += 1
            #     elif pred == 0 and true_val == 0:
            #         target_tn += 1
            #     elif pred == 0 and true_val == 1:
            #         target_fn += 1
            #     print(f'{name}  {j} {true_val}: tp{target_tp}, fp{target_fp}, tn{target_tn}, fn{target_fn}')
            # else:
            #         # 更新混淆矩阵
            #     if pred == 1 and true_val == 1:
            #         other_tp += 1
            #     elif pred == 1 and true_val == 0:
            #         other_fp += 1
            #     elif pred == 0 and true_val == 0:
            #         other_tn += 1
            #     elif pred == 0 and true_val == 1:
            #         other_fn += 1
                # print(f'{name}  {j}: {true_val}')

            # 计算特定碱基对的指标
            # target_accuracy = target_correct / target_total if target_total > 0 else 0
            # target_avg_loss = target_total_loss / target_total if target_total > 0 else 0
            target_precision = target_tp / (target_tp + target_fp) if (target_tp + target_fp) > 0 else 0
            target_recall = target_tp / (target_tp + target_fn) if (target_tp + target_fn) > 0 else 0
            target_f1_score = (2 * target_tp) / (2 * target_tp + target_fp + target_fn) if (target_tp + target_fn) > 0 else 0

            # 计算其他碱基对的指标
            # other_accuracy = other_correct / other_total if other_total > 0 else 0
            # other_avg_loss = other_total_loss / other_total if other_total > 0 else 0
            other_precision = other_tp / (other_tp + other_fp) if (other_tp + other_fp) > 0 else 0
            other_recall = other_tp / (other_tp + other_fn) if (other_tp + other_fn) > 0 else 0
            other_f1_score = (2 * other_tp) / (2 * other_tp + other_fp + other_fn) if (other_tp + other_fn) > 0 else 0

            # 计算MCC
            target_mcc_numerator = (target_tp * target_tn) - (target_fp * target_fn)
            target_mcc_denominator = ((target_tp + target_fp) * (target_tp + target_fn) *
                                      (target_tn + target_fp) * (target_tn + target_fn)) ** 0.5
            target_mcc = target_mcc_numerator / target_mcc_denominator if target_mcc_denominator > 0 else 0

            other_mcc_numerator = (other_tp * other_tn) - (other_fp * other_fn)
            other_mcc_denominator = ((other_tp + other_fp) * (other_tp + other_fn) *
                                     (other_tn + other_fp) * (other_tn + other_fn)) ** 0.5
            other_mcc = other_mcc_numerator / other_mcc_denominator if other_mcc_denominator > 0 else 0
        # else:
        #     print('no target_precision')
    return target_precision,target_recall,target_f1_score,target_mcc,other_precision,other_recall,other_f1_score,other_mcc

# randomly select one sample from the test set and perform the evaluation

def model_eval_all_test(contact_net, test_generator):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    result_nc = list()
    result_nc_tmp = list()
    ct_dict_all = {}
    dot_file_dict = {}
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []

    pes_precision_list = []
    pes_recall_list = []
    pes_f1_list = []
    pes_mcc_list = []

    other_precision_list = []
    other_recall_list = []
    other_f1_list = []
    other_mcc_list = []

    # 在循环开始前初始化列表
    pes_detailed_results = []
    other_detailed_results = []

    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight)
    # for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        # for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_merge_generator:
        # for contacts, seq_embeddings,seq_embeddings_1, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
        # pdb.set_trace()
        nc_map_nc = nc_map.float() * contacts
        if seq_lens.item() > 1500:
            continue
        if batch_n % 1000 == 0:
            print('Batch number: ', batch_n)
        # if batch_n > 3:
        batch_n += 1
        #    break
        # if batch_n-1 in rep_ind:
        #    continue
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        # matrix_reps_batch = torch.unsqueeze(
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())
        #     torch.Tensor(matrix_reps.float()).to(device), -1)

        # state_pad = torch.zeros([matrix_reps_batch.shape[0],
        #     seq_len, seq_len]).to(device)

        # PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        tik = time.time()

        with torch.no_grad():
            # pred_contacts = contact_net(seq_embedding_batch,seq_embedding_batch_1)
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
                                 seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)  ## 1.6
        # seq_ori, 0.01, 0.1, 100, 1.6, True) ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()

        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)

        print('map_no_train',map_no_train.shape)
        target_precision,target_recall,target_f1_score,target_mcc,other_precision,other_recall,other_f1_score,other_mcc= evaluate_model_separate_bp(map_no_train.cpu()[0],contacts_batch.cpu()[0], seq_name)

        pes_precision_list.append(target_precision)
        pes_recall_list.append(target_recall)
        # acc_list.append(acc.item())
        pes_f1_list.append(target_f1_score)
        pes_mcc_list.append(target_mcc)

        other_precision_list.append(other_precision)
        other_recall_list.append(other_recall)
        other_f1_list.append(other_f1_score)
        other_mcc_list.append(other_mcc)

        # 记录每个样本的指标 - 表格格式，空格分隔
        sample_name = seq_name[0] if isinstance(seq_name, list) else seq_name
        pes_detailed_results.append(
            f"{sample_name} {target_precision:.4f} {target_recall:.4f} {target_f1_score:.4f} {target_mcc:.4f}")
        other_detailed_results.append(
            f"{sample_name} {other_precision:.4f} {other_recall:.4f} {other_f1_score:.4f} {other_mcc:.4f}")

    target_metrics = (np.mean(pes_f1_list), np.mean(pes_precision_list), np.mean(pes_recall_list),
                      np.mean(pes_mcc_list))
    other_metrics = (np.mean(other_f1_list), np.mean(other_precision_list), np.mean(other_recall_list),
                     np.mean(other_mcc_list))

    target_F1, target_preci, target_recall, target_mcc = target_metrics
    other_F1, other_preci, other_recall, other_mcc = other_metrics

    # 写入Target Base Pairs指标到文件
    output_file_target = '/home/chenjingjing/Models/UFold/UFold/models/pes/record/target_base_pairs_metrics.txt'
    os.makedirs(os.path.dirname(output_file_target), exist_ok=True)
    with open(output_file_target, 'w') as f:
        for result in pes_detailed_results:
            f.write(result + '\n')

    # 写入Other Base Pairs指标到文件
    output_file_other = '/home/chenjingjing/Models/UFold/UFold/models/pes/record/other_base_pairs_metrics.txt'
    os.makedirs(os.path.dirname(output_file_other), exist_ok=True)
    with open(output_file_other, 'w') as f:
        for result in other_detailed_results:
            f.write(result + '\n')

    print("\n=== Target Base Pairs Results ===")
    print("test_preci=%.5f, test_recall=%.5f, test_F1=%.5f, test_MCC=%.5f" % (
        target_preci, target_recall, target_F1, target_mcc))

    print("\n=== Other Base Pairs Results ===")
    print("test_preci=%.5f, test_recall=%.5f, test_F1=%.5f, test_MCC=%.5f" % (
        other_preci, other_recall, other_F1, other_mcc))
    #
    # nt_exact_p, nt_exact_r, nt_exact_f1, nt_exact_mcc = zip(*result_no_train)
    # # pdb.set_trace()
    # print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    # print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    # print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
    # print('Average testing MCC with pure post-processing: ', np.average(nt_exact_mcc))

    # with open('/data2/darren/experiment/ufold/results/sample_result.pickle','wb') as f:
    #    pickle.dump(result_dict,f)
    # with open('../results/rnastralign_short_pure_pp_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(0)

    # pdb.set_trace()

    config_file = args.config
    test_file = args.test_files

    config = process_config(config_file)
    # print('Here is the configuration of this run: ')
    # print(config)

    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet30.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_reduce_noupscale25.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_reduce15.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_noupscale90.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_noupscale_simfam90.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_upscale7fam_90.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_ori49.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_8dim49.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar17.pt' #unet_bpTR0_addsimmutate_addmoresimilar48.pt
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0_33.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_RNAlign_49.pt'
    if test_file in ['TS1', 'TS2', 'TS3']:
        MODEL_SAVED = 'models/ufold_train.pt'
    elif test_file == 'Rivas_A' or test_file == 'TestSetB':
        MODEL_SAVED = '/home/chenjingjing/Models/UFold/UFold/models/Rivas/ufold_train_99.pt'
    elif test_file == 'RNAStralign' or test_file == 'ArchiveII':
        MODEL_SAVED = '/home/chenjingjing/Models/UFold/UFold/models/RNAStrAlign/ufold_train_99.pt'
    elif test_file == 'bprna_1m' or test_file == 'bprna_new' or test_file == 'bprna_pes':
        MODEL_SAVED = '/home/chenjingjing/Models/UFold/UFold/models/bprna_1m/ufold_train_99.pt'
    elif test_file in ['16s','5s','23s','grp1','grp2','RNaseP','srp','telomerase','tmRNA','tRNA']:
        MODEL_SAVED = f'/home/chenjingjing/Models/UFold/UFold/models/crossfamily_new/{test_file}/ufold_train_99.pt'
        # MODEL_SAVED = f'/home/chenjingjing/Models/UFold/UFold/models/crossfamily_new/5s/ufold_train_99.pt'
    elif test_file in ['short','medium']:
        MODEL_SAVED = f'/home/chenjingjing/Models/UFold/UFold/models/length/ufold_train_99.pt'
    elif test_file == 'pes':
        MODEL_SAVED = f'/home/chenjingjing/Models/UFold/UFold/models/pes/ufold_train_99.pt'
    else:
        MODEL_SAVED = 'models/ufold_train.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0andMXUnet_99.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0bpnewOriuseMXUnet_96.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0_extract_99.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0_continuefrom99_56.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar_finetune0.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar_twochannel48.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar_25dim46.pt'
    # MODEL_SAVED = '/data2/darren/experiment/models_ckpt/concat_trainsetA/unet_trainsetA14.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_upscaleonlyPDB_outer7.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_upscaleallfams_19.ptunet_rfam12_upscaleonlyPDB_finetune8.pt'
    # MODEL_SAVED = '/data2/darren/experiment/models_ckpt/concat/'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpRNA26.pt'
    # MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet48.pt'

    # os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu

    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    model_path = '/data2/darren/experiment/ufold/models_ckpt/'.format(model_type, data_type, d)
    epoches_first = config.epoches_first

    # if gpu is to be used
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    seed_torch()

    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that

    # train_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'train', True)
    # val_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'val')
    ##test_data = RNASSDataGenerator('./data/{}/'.format(data_type), 'test_no_redundant.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant_600.pickle')
    print('Loading test file: ', test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('/home/chenjingjing/Models/UFold/UFold/data/ArchiveII/', test_file + '.pickle')
    elif test_file == 'Rivas_A' or test_file == 'TestSetB':
        test_data = (RNASSDataGenerator('/home/chenjingjing/Models/UFold/UFold/data/Rivas/', test_file + '.cPickle'))
    elif test_file == 'bprna_1m':
        test_data = RNASSDataGenerator('/home/chenjingjing/Models/UFold/UFold/data/bprna_1m/', 'test.pickle')
    elif test_file == 'bprna_new':
        test_data = RNASSDataGenerator('/home/chenjingjing/Models/E2Efold/data/bprna_new/', 'bprna_new.pickle')
    elif test_file in ['16s','5s','23s','grp1','grp2','RNaseP','srp','telomerase','tmRNA','tRNA']:
        test_data = RNASSDataGenerator(f'/home/chenjingjing/Models/UFold/UFold/data/crossfamily_new/{test_file}/', 'test.pickle')
        # test_data = RNASSDataGenerator('/home/chenjingjing/Models/UFold/UFold/data/crossfamily_new/5s/',
        #                                'test.pickle')
    elif test_file in ['short','medium']:
        test_data = RNASSDataGenerator(f'/home/chenjingjing/Models/UFold/UFold/data/length/', test_file + '.pickle')
    elif test_file == 'pes' or test_file == 'bprna_pes':
        test_data = RNASSDataGenerator('/home/chenjingjing/Models/UFold/UFold/data/pes/', 'test.pickle')
    else:
        test_data = RNASSDataGenerator('data/', test_file + '.cPickle')

    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'all_1800_archieveII.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'all_600_archieveII.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_test_generate.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_RF00001_similarfamilys_test.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TestSetA.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_allfamily_generate_test.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_38family_generate_test.cPickle')
    ##test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_new_generate.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_new20201015.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'archieveII_contacts_pred.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_pdbnewgenerate_yingxc.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TS0_ori.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TR0_leavefortest1000.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TR0_andsim_mutate_extract_train.cPickle')
    seq_len = test_data.data_y.shape[-2]
    print('Max seq length ', seq_len)
    # pdb.set_trace()

    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    # # train_set = Dataset(train_data)
    # train_set = Dataset_FCN(train_data)
    # train_generator = data.DataLoader(train_set, **params)

    # # val_set = Dataset(val_data)
    # val_set = Dataset_FCN(val_data)
    # val_generator = data.DataLoader(val_set, **params)

    # test_set = Dataset(test_data)
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)

    '''
    test_merge = Dataset_FCN_merge(test_data,test_data2)
    test_merge_generator = data.DataLoader(test_merge, **params)
    pdb.set_trace()
    '''

    contact_net = FCNNet(img_ch=17)

    # pdb.set_trace()
    print('==========Start Loading==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location='cuda:1'))
    print('==========Finish Loading==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    model_eval_all_test(contact_net, test_generator)

    # if LOAD_MODEL and os.path.isfile(model_path):
    #     print('Loading u net model...')
    #     contact_net.load_state_dict(torch.load(model_path))

    # u_optimizer = optim.Adam(contact_net.parameters())


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    main()