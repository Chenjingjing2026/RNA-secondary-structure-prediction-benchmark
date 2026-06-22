from .dataset import RNADataset
from torch.utils.data import DataLoader


# def load_data(data_name, batch_size, data_root, num_workers=8, **kwargs):
#     if data_name == 'RNAStralign':
#         test_set = RNADataset(path=data_root, dataname='RNAStrAlign')
#     elif data_name == 'ArchiveII':
#         test_set = RNADataset(path=data_root, dataname='all_600')
#     elif data_name == 'bpRNA':
#         test_set = RNADataset(path=data_root, dataname='test')
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
#     return test_loader


def load_data(data_name, batch_size, data_root, num_workers=8, **kwargs):
    if data_name == 'RNAStralign':
        train_set = RNADataset(path=data_root, dataname='train')
        val_set = RNADataset(path=data_root, dataname='valid')
        test_set = RNADataset(path=data_root, dataname='test')
    elif data_name == 'Rivas':
        train_set = RNADataset(path=data_root, dataname='TrainSetA')
        val_set = RNADataset(path=data_root, dataname='TestSetA')
        test_set = RNADataset(path=data_root, dataname='TestSetB')
    elif data_name == 'bpRNA':
        train_set = RNADataset(path=data_root, dataname='train')
        val_set = RNADataset(path=data_root, dataname='valid')
        test_set = RNADataset(path=data_root, dataname='test')
    elif data_name == 'bprna_new':
        train_set = RNADataset(path=data_root, dataname='bprna_new')
        val_set = RNADataset(path=data_root, dataname='bprna_new')
        test_set = RNADataset(path=data_root, dataname='bprna_new')
    elif data_name in ['16s','5s','23s','grp1','grp2','RNaseP','srp','telomerase','tmRNA','tRNA']:
        # data_root="/home/chenjingjing/Models/RFold/RFold-master/mydata/crossfamily/except_16s"
        train_set = RNADataset(path=data_root, dataname='train')
        val_set = RNADataset(path=data_root, dataname='valid')
        test_set = RNADataset(path=data_root, dataname='test')
    elif data_name == 'short':
        # data_root="/home/chenjingjing/Models/RFold/RFold-master/mydata/crossfamily/except_16s"
        train_set = RNADataset(path=data_root, dataname='train')
        val_set = RNADataset(path=data_root, dataname='valid')
        test_set = RNADataset(path=data_root, dataname='short')
    elif data_name == 'medium':
        # data_root="/home/chenjingjing/Models/RFold/RFold-master/mydata/crossfamily/except_16s"
        train_set = RNADataset(path=data_root, dataname='train')
        val_set = RNADataset(path=data_root, dataname='valid')
        test_set = RNADataset(path=data_root, dataname='medium')
    elif data_name == 'pes':
        # data_root="/home/chenjingjing/Models/RFold/RFold-master/mydata/crossfamily/except_16s"
        train_set = RNADataset(path=data_root, dataname='train')
        val_set = RNADataset(path=data_root, dataname='valid')
        test_set = RNADataset(path=data_root, dataname='test')

    # 创建三个 DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader, test_loader
