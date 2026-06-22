import json
import torch
import logging
import collections
import os.path as osp
from parser import create_parser

import warnings

warnings.filterwarnings('ignore')

from utils import *
from rfold import RFold



class Exp:
    def __init__(self, args):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        self._get_data()  # 确保这个方法同时加载训练数据
        print_log(output_namespace(self.args))

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:2')
            print('Use GPU:', device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        self.method = RFold(self.args, self.device)

    def _get_data(self):
        self.test_loader = get_dataset(self.config)

    def test(self):
        # 如果指定了测试模式且提供了模型路径，则加载模型
        if self.args.test_only and self.args.model_path:
            self.method.model.load_state_dict(torch.load(self.args.model_path))
            self.method.model.eval()

        test_f1, test_precision, test_recall, test_mcc, test_runtime = self.method.test_one_epoch(self.test_loader)
        print_log('Test F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, MCC: {3:.4f}, Runtime: {4:.4f}\n'.format(
            test_f1, test_precision, test_recall, test_mcc, test_runtime))
        return test_f1, test_precision, test_recall, test_mcc, test_runtime

    def _get_data(self):
        # 修改为同时加载训练数据
        self.train_loader, self.val_loader, self.test_loader = get_dataset(self.config)

    def train(self):
        # 如果是仅测试模式，则跳过训练
        if self.args.test_only:
            print_log("Test-only mode, skipping training")
            return

        best_f1 = 0
        for epoch in range(self.args.epoch):
            train_loss, train_f1, train_precision, train_recall, train_mcc, train_time = self.method.train_one_epoch(
                self.train_loader)
            val_f1, val_precision, val_recall, val_mcc, _ = self.method.test_one_epoch(self.val_loader)

            # 打印日志
            print_log(f'Epoch {epoch + 1}/{self.args.epoch}')
            print_log(f'Train Loss: {train_loss:.4f}')
            print_log(
                f'Val F1: {val_f1:.4f} Precision: {val_precision:.4f} Recall: {val_recall:.4f} MCC: {val_mcc:.4f}')

            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.method.model.state_dict(), f'{self.checkpoints_path}/best_model.pth')


if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

    args = create_parser()
    config = args.__dict__
    exp = Exp(args)

    if not args.test_only:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.train()
    # exp.train()  # 需要先实现train方法
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    test_f1, test_precision, test_recall, test_mcc, test_runtime = exp.test()