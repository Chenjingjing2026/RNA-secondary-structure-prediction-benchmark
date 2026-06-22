# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common.utils import add_parent_path
from common.experiment import add_exp_args as add_exp_args_parent
from common.experiment import DiffusionExperiment
from common.data_utils import contact_map_masks
from common.loss_utils import bce_loss, evaluate_f1_precision_recall
from common.loss_utils import calculate_auc, calculate_mattews_correlation_coefficient,rna_evaluation,rna_evaluation_modified
add_parent_path(level=2)

TEST=False

def add_exp_args(parser):
    add_exp_args_parent(parser)

def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device,non_blocking=True)
        # return batch.to(device)
    elif isinstance(batch, dict):
        return {key: batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(item, device) for item in batch]
    else:
        return batch

class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        device = self.args.device
        total = len(self.train_loader)
        # for _, (contact, data_fcn_2, data_seq_raw, data_length, _, set_max_len, data_seq_encoding) in enumerate(self.train_loader):
        for idx, batch in enumerate(self.train_loader):
            batch=batch_to_device(batch, device)
            self.optimizer.zero_grad()
            matrix_rep = torch.zeros_like(batch['contact'])
            # contact = contact.to(device)
            # data_fcn_2 = data_fcn_2.to(device)
            # data_length = data_length.to(device)
            # data_seq_raw = data_seq_raw.to(device)
            # data_seq_encoding = data_seq_encoding.to(device)
            contact_masks = contact_map_masks(batch['data_length'], matrix_rep).to(device)  # data_length以内的为1
            contact_masks = contact_masks.unsqueeze(1)
            # loss = self.model(contact, data_fcn_2, data_seq_raw, contact_masks, set_max_len, data_seq_encoding)
            loss = self.model(batch['contact'], batch['data_fcn_2'], batch['tokens'],contact_masks, batch['set_max_len'], batch['data_seq_encode_pad'])
            loss.backward()

            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(batch['contact'])
            loss_count += len(batch['contact'])
            print('Training. Epoch: {}/{},Iter:{}/{}, Bits/dim: {:.5f}'.
                  format(epoch + 1, self.args.epochs,idx,total, loss_sum / loss_count), end='\r')
            if TEST:
                break
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        from datetime import datetime
        torch.save(self.model.state_dict(), f'/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/crossfamily_short/tRNA/backup/ckpt_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_{epoch}.pth')
        return {'bpd': loss_sum / loss_count}

    def val_fn(self, epoch):
        self.model.eval()

        device = self.args.device
        with torch.no_grad():
            loss_count = 0
            val_loss_sum = 0.0
            auc_score = 0.0
            auc_count = 0
            val_no_train = list()
            mcc_no_train = list()

            for _, batch in enumerate(self.val_loader):
                batch=batch_to_device(batch, device)
                self.optimizer.zero_grad()
                matrix_rep = torch.zeros_like(batch['contact'])
                # contact = contact.to(device)
                # data_fcn_2 = data_fcn_2.to(device)
                # data_length = data_length.to(device)
                # data_seq_raw = data_seq_raw.to(device)
                # data_seq_encoding = data_seq_encoding.to(device)
                contact_masks = contact_map_masks(batch['data_length'], matrix_rep).to(device)  # data_length以内的为1
                contact_masks = contact_masks.unsqueeze(1)
                # calculate contact loss
                batch_size = batch['contact'].shape[0]
                pred_x0, _ = self.model.sample(batch_size, batch['data_fcn_2'], batch['tokens'], batch['set_max_len'], contact_masks,batch['data_seq_encode_pad'])

                # pred_x0, _ = self.model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)
                contact = batch['contact'].cpu()
                pred_x0 = pred_x0.cpu().float()

                val_loss_sum += bce_loss(pred_x0.float(), contact.float()).cpu().item()
                loss_count += len(contact)
                # print(contact.shape)
                # print(contact.unique())
                try:
                    auc_score += roc_auc_score(contact.reshape(-1).cpu().numpy(), pred_x0.reshape(-1).cpu().numpy(),
                                               average='micro')
                except ValueError:
                    auc_score += 0  # 或者跳过此项
                    print("由于 y_true 中只有一个类别，AUC 计算被跳过")
                # auc_score += calculate_auc(contact.float(), pred_x0)
                auc_count += 1
                val_no_train_tmp = list(map(lambda i: evaluate_f1_precision_recall(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                val_no_train += val_no_train_tmp

                mcc_no_train_tmp = list(map(lambda i: calculate_mattews_correlation_coefficient(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                mcc_no_train += mcc_no_train_tmp
                # if TEST:
                #     break

            val_precision, val_recall, val_f1 = zip(*val_no_train)

            val_precision = np.average(np.nan_to_num(np.array(val_precision)))
            val_recall = np.average(np.nan_to_num(np.array(val_recall)))
            val_f1 = np.average(np.nan_to_num(np.array(val_f1)))

            mcc_final = np.average(np.nan_to_num(np.array(mcc_no_train)))

            print('#' * 80)
            print('Average val F1 score: ', round(val_f1, 3))
            print('Average val precision: ', round(val_precision, 3))
            print('Average val recall: ', round(val_recall, 3))
            print('#' * 80)
            print('Average val MCC', round(mcc_final, 3))
            print('#' * 80)
            print('')
        return {'f1': val_f1, 'precision': val_precision, 'recall': val_recall,
                'auc_score': auc_score / auc_count, 'mcc': mcc_final, 'bce_loss': val_loss_sum / loss_count}

    def test_fn(self, epoch=0):
        self.model.eval()
        device = self.args.device
        with torch.no_grad():
            test_no_train = list()
            total_name_list = list()
            total_length_list = list()

            detailed_results_core =[]

            for _, batch in enumerate(
                    self.test_loader):
                batch=batch_to_device(batch, device)
                total_name_list += [item for item in batch['data_name']]
                total_length_list += [item.item() for item in batch['data_length']]
                matrix_rep = torch.zeros_like(batch['contact'])
                # contact = contact.to(device)
                # data_fcn_2 = data_fcn_2.to(device)
                # data_length = data_length.to(device)
                # data_seq_raw = data_seq_raw.to(device)
                # data_seq_encoding = data_seq_encoding.to(device)
                contact_masks = contact_map_masks(batch['data_length'], matrix_rep).to(device)  # data_length以内的为1
                contact_masks = contact_masks.unsqueeze(1)
                # calculate contact loss
                batch_size = batch['contact'].shape[0]
                pred_x0, _ = self.model.sample(batch_size, batch['data_fcn_2'], batch['tokens'], batch['set_max_len'], contact_masks,batch['data_seq_encode_pad'])

                # data_fcn_2 = data_fcn_2.to(device)
                # matrix_rep = torch.zeros_like(batch['contact'])
                # data_length = data_length.to(device)
                # data_seq_raw = data_seq_raw.to(device)
                # data_seq_encoding = data_seq_encoding.to(device)
                # contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

                # calculate contact loss
                # batch_size = batch['contact'].shape[0]
                # pred_x0, _ = self.model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)

                pred_x0 = pred_x0.cpu().float()

                # test_no_train_tmp = list(map(lambda i: rna_evaluation(  #orig
                #     pred_x0[i].squeeze(), batch['contact'].cpu().float()[i].squeeze()), range(pred_x0.shape[0])))
                test_no_train_tmp = list(map(lambda i: rna_evaluation(
                    pred_x0[i].squeeze(), batch['contact'].cpu().float()[i].squeeze()), range(pred_x0.shape[0])))
                test_no_train += test_no_train_tmp
                # if TEST:
                #     break

                # 记录每个样本的核心指标 - 一行格式
                for i, (acc, prec, rec, sens, spec, f1, mcc) in enumerate(test_no_train_tmp):
                    sample_name = total_name_list[i] if i < len(total_name_list) else f"sample_{i}"
                    detailed_results_core.append(f"{sample_name} {prec:.4f} {rec:.4f} {f1:.4f} {mcc:.4f}")

            # 写入核心指标
            output_file_core = '/home/chenjingjing/Models/RNA_DiffFold/RNADiffFold/result/pes1/detailed_metrics_core.txt'
            with open(output_file_core, 'w') as f:
                f.write("Name Precision Recall F1 MCC\n")
                for result in detailed_results_core:
                    f.write(result + '\n')

            accuracy, prec, recall, sens, spec, F1, MCC = zip(*test_no_train)

            f1_pre_rec_df = pd.DataFrame({'name': total_name_list,
                                          'length': total_length_list,
                                          'accuracy': list(np.array(accuracy)),
                                          'precision': list(np.array(prec)),
                                          'recall': list(np.array(recall)),
                                          'sensitivity': list(np.array(sens)),
                                          'specificity': list(np.array(spec)),
                                          'f1': list(np.array(F1)),
                                          'mcc': list(np.array(MCC))})

            accuracy = np.average(np.nan_to_num(np.array(accuracy)))
            precision = np.average(np.nan_to_num(np.array(prec)))
            recall = np.average(np.nan_to_num(np.array(recall)))
            sensitivity = np.average(np.nan_to_num(np.array(sens)))
            specificity = np.average(np.nan_to_num(np.array(spec)))
            F1 = np.average(np.nan_to_num(np.array(F1)))
            MCC = np.average(np.nan_to_num(np.array(MCC)))

            print('#' * 40)
            print('Average testing accuracy: ', round(accuracy, 5))
            print('Average testing F1 score: ', round(F1, 5))
            print('Average testing precision: ', round(precision, 5))
            print('Average testing recall: ', round(recall, 5))
            print('Average testing sensitivity: ', round(sensitivity, 5))
            print('Average testing specificity: ', round(specificity, 5))
            print('#' * 40)
            print('Average testing MCC', round(MCC, 5))
            print('#' * 40)
            print('')
        return {'f1': F1, 'precision': precision, 'recall': recall,
                'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy, 'mcc': MCC}, f1_pre_rec_df
