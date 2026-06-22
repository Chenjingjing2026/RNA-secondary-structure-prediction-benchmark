import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='/home/chenjingjing/Models/RFold/RFold-master/results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)

    # dataset parameters
    parser.add_argument('--data_name', default='Rivas', choices=['Rivas', 'RNAStralign', 'bpRNA','bprna_new','bprna_1m','bprna_new','16s','5s','23s','grp1','grp2','RNaseP','srp','telomerase','tmRNA','tRNA','short','medium','pes'])
    parser.add_argument('--data_root', default='./data/archiveII_all')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # debug parameters
    parser.add_argument('--num_hidden', default=128, type=int)
    parser.add_argument('--pf_dim', default=128, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)

    # Testing parameters (新增部分)
    parser.add_argument('--test_only', action='store_true', help='only test without training')
    parser.add_argument('--model_path', default='', type=str,
                        help='path to pretrained model for testing (required when test_only=True)')
    return parser.parse_args()