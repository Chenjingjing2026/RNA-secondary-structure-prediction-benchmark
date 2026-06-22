# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import input_data
import model
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef


N_CLASSES = 3
IMG_W = 200
IMG_H = 19
BATCH_SIZE = 1
CAPACITY = 2000


def evaluate():
    all_true_labels = []  # 初始化空列表
    all_predictions = []
    with tf.Graph().as_default():
        # logs_train_dir = '/home/zhangch/RSS/log/'
        # verification_dir = "/home/zhangch/Desktop/5S_verset/"
        logs_train_dir = '/home/chenjingjing/Models/CDPfold/results/bprna_1m/short/'
        verification_dir = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/test_set/'
        n_test = 22299
        train,train_label = input_data.get_files(verification_dir)
        train_batch,train_label_batch = input_data.get_batch(train,
                                                         train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
        logit = model.inference(train_batch,BATCH_SIZE,N_CLASSES) 
        top_k_op = tf.nn.in_top_k(logit,train_label_batch,1)       
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("Loading success,global_step is %s" % global_step)
            else:
                print("no checkpoint file found")
                
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord = coord)
            
            try:
                num_iter = int(n_test/BATCH_SIZE)
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
                
                # while step < num_iter and not coord.should_stop():
                #     prediction = sess.run([top_k_op])
                #     true_count += np.sum(prediction)
                #     step += 1
                #     precision = float(true_count)/total_sample_count
                # print("precision = %3f"%precision)
                
                while step < num_iter and not coord.should_stop():
                    prediction, labels = sess.run([top_k_op, train_label_batch])
                    true_count += np.sum(prediction)
                    all_true_labels.extend(labels)  # 收集真实标签
                    all_predictions.extend(prediction)  # 收集预测标签
                    step += 1

                # 将列表转换为 NumPy 数组
                all_true_labels = np.array(all_true_labels)
                all_predictions = np.array(all_predictions)

                # 类别数量
                num_classes = len(np.unique(all_true_labels))

                 # 初始化 TP, FP, FN
                TP = np.zeros(num_classes)
                FP = np.zeros(num_classes)
                FN = np.zeros(num_classes)
                TN = np.zeros(num_classes)

                # 计算 TP, FP, FN, TN
                for i in range(num_classes):
                    TP[i] = np.sum((all_true_labels == i) & (all_predictions == i))
                    FP[i] = np.sum((all_true_labels != i) & (all_predictions == i))
                    FN[i] = np.sum((all_true_labels == i) & (all_predictions != i))
                    TN[i] = np.sum((all_true_labels != i) & (all_predictions != i))
                    # 计算 TP, FP, FN
                # for true_label, pred_label in zip(all_true_labels, all_predictions):
                #     if true_label == pred_label:
                #         TP[true_label] += 1  # True Positive
                #     else:
                #         FP[pred_label] += 1  # False Positive
                #         FN[true_label] += 1  # False Negative

                 # 计算 precision, recall, F1 分数
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)

                # 处理除零情况（如果某个类别没有样本）
                precision = np.nan_to_num(precision, nan=0.0)
                recall = np.nan_to_num(recall, nan=0.0)
                f1 = np.nan_to_num(f1, nan=0.0)

                # 计算 MCC
                mcc_numerator = TP * TN - FP * FN
                mcc_denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                mcc = mcc_numerator / mcc_denominator
                mcc = np.nan_to_num(mcc, nan=0.0)  # 处理除零情况
                # 打印每个类别的结果
                # for i in range(num_classes):
                #     print(f"Class {i}:")
                #     print(f"  Precision: {precision[i]:.4f}")
                #     print(f"  Recall: {recall[i]:.4f}")
                #     print(f"  F1 Score: {f1[i]:.4f}")
                #     print()

                # 计算宏平均（Macro Average）
                macro_precision = np.mean(precision)
                macro_recall = np.mean(recall)
                macro_f1 = np.mean(f1)
                macro_mcc = np.mean(mcc)

                print(f"Macro Precision: {macro_precision:.5f}")
                print(f"Macro Recall: {macro_recall:.5f}")
                print(f"Macro F1 Score: {macro_f1:.5f}")
                print(f"Macro MCC: {macro_mcc:.5f}")

                # # 将列表转换为 NumPy 数组
                # all_true_labels = np.array(all_true_labels)
                # all_predictions = np.array(all_predictions)
                
                # precision = precision_score(all_true_labels, all_predictions)
                # recall = recall_score(all_true_labels, all_predictions)
                # f1 = f1_score(all_true_labels, all_predictions)
                # mcc = matthews_corrcoef(np.array(all_true_labels), np.array(all_predictions))
                #
                # print(f"Precision = {precision:.4f}")
                # print(f"Recall = {recall:.4f}")
                # print(f"F1 Score = {f1:.4f}")
                # print(f"MCC = {mcc:.4f}")

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

evaluate()

