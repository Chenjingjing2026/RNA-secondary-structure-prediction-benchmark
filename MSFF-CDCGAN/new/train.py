import torch
import os
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from mydataset import MyDataset, transform, target_transform
import sys
from tqdm import tqdm
sys.path.append('/home/chenjingjing/Models/MSFF-CDCGAN/new')

from msff_cdcgan_model import G_cdcgan, D_cdcgan
from auxiliary_function import calculation, Structure_matrix, denorm, Correction_unit
import sklearn.metrics as metrics
from visdom import Visdom
import math
from sklearn.model_selection import KFold
pd.set_option('display.expand_frame_repr', False)  #
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码

np.set_printoptions(threshold=np.inf)
import os

# 设置环境变量，指定可用的 GPU（例如 GPU 0, 1, 2）
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

#######################################################################################################################

###### 基本参数
batch_size = 32
total_epochs = 10
G_lr = 0.01
D_lr = 0.02

path = 'bprna_new'
# data_path ='/home/chenjingjing/Models/MSFF-CDCGAN/lable/RNAStrAlign/train/train-data.txt'
# data_path = '/home/yuanshuai20/RNAs/lable/totla.txt'
real_img_path = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/real_img'
fake_img_path = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/fake_img'
real_seq_path = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/real_seq'
fake_seq_path = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/fake_seq'
if not os.path.exists(real_img_path):
    os.makedirs(real_img_path)
if not os.path.exists(fake_img_path):
    os.makedirs(fake_img_path)
if not os.path.exists(real_seq_path):
    os.makedirs(real_seq_path)
if not os.path.exists(fake_seq_path):
    os.makedirs(fake_seq_path)
Train_gloss = open('/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/Train_gloss.txt', 'w')
Train_dloss = open('/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/Train_dloss.txt', 'w')
Train_acc = open('/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/Train_acc.txt', 'w')
Test_gloss = open('/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/Test_gloss.txt', 'w')
Test_dloss = open('/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/Test_dloss.txt', 'w')
Test_acc = open('/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/Test_acc.txt', 'w')
####### 划分数据集

# custom_dataset = MyDataset(txt=data_path, transform=transform, target_transform=target_transform)
# train_size = int(len(custom_dataset) * 0.6)
# validate_size = int(len(custom_dataset) * 0.2)
# test_size = len(custom_dataset) - train_size - validate_size
# train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,
#                                                                               [train_size, validate_size, test_size])
# print(
#     "train_dataset_size:", train_size, "validate_dataset_size:", validate_size, "test_dataset_size", test_size)
# 1. 读取指定的文件路径
train_data_path = '/home/chenjingjing/Models/MSFF-CDCGAN/lable/bprna_1m/train/Filtered_train-data.txt'
validate_data_path = '/home/chenjingjing/Models/MSFF-CDCGAN/lable/bprna_1m/valid/Filtered_valid-data.txt'
test_data_path = '/home/chenjingjing/Models/MSFF-CDCGAN/lable/bprna_new/test/Filtered_test-data.txt'

# 2. 创建数据集
train_dataset = MyDataset(txt=train_data_path, transform=transform, target_transform=target_transform)
validate_dataset = MyDataset(txt=validate_data_path, transform=transform, target_transform=target_transform)
test_dataset = MyDataset(txt=test_data_path, transform=transform, target_transform=target_transform)

# 3. 打印数据集的大小
print("train_dataset_size:", len(train_dataset))
print("validate_dataset_size:", len(validate_dataset))
print("test_dataset_size:", len(test_dataset))

print(
    "train_dataset:", train_dataset, "validate_dataset:", validate_dataset, "test_dataset", test_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True, num_workers=16,
                             drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# 初始化构建判别器和生成器
discriminator = D_cdcgan()
generator = G_cdcgan()

discriminator.weight_init(mean=0.0, std=0.02)
generator.weight_init(mean=0.0, std=0.02)

# 多GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    discriminator = torch.nn.DataParallel(discriminator)
    generator = torch.nn.DataParallel(generator)

discriminator = discriminator.to(device)
generator = generator.to(device)

# 初始化二值交叉熵损失
bce = torch.nn.BCELoss().to(device)

# 初始化优化器，使用Adam优化器
g_optimizer = optim.Adam(generator.parameters(), lr=G_lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=D_lr, betas=(0.5, 0.999))
scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.1, min_lr=0.0001)
scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.1, min_lr=0.0002)

##固定种子
torch.manual_seed(23)
np.random.seed(23)
torch.cuda.manual_seed_all(23)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def visdom(epoch, tg, td, vg, vd, ta, va):
    viz = Visdom(env='Train and val loss')
    if (epoch + 1) % 50 == 0:
        viz.line([[ta, va]], [epoch], win='acc',
                 opts=dict(title='acc', legend=['train acc', 'val acc']), update='append')
    viz.line([[0, 0]], [0], win='train_loss', opts=dict(title='train loss', legend=['gloss', 'dloss']))
    viz.line([[tg, td]], [epoch], win='train_loss',
             opts=dict(title='train loss', legend=['gloss', 'dloss']), update='append')
    viz.line([[0, 0]], [0], win='val_loss', opts=dict(title='val loss', legend=['gloss', 'dloss']))
    viz.line([[vg, vd]], [epoch], win='val_loss', opts=dict(title='val loss', legend=['gloss', 'dloss']),
             update='append')


# 训练集
def Train_dataset(g_model, d_model, Structure_matrix, calculation):
    g_model.train()
    d_model.train()
    train_gloss = []
    train_dloss = []
    TP_total = []
    TN_total = []
    FP_total = []
    FN_total = []

    for i, data in enumerate(train_loader):
        ###################数据准备####################
        real_Seq_images, real_Str_images, RNAlen, name, rlabel = data
        # print('real_Seq_images, real_Str_images, RNAlen, name, rlabel',real_Seq_images, real_Str_images, RNAlen, name, rlabel)
        real_Str_images = real_Str_images.to(device)
        c = real_Seq_images.to(device)
        z = torch.randn([batch_size, c.size(1), c.size(2), c.size(3)]).to(device)  # 噪声

        ###################训练D####################
        fake_Str_images = generator(c, z)
        # print('fake_Str_images',fake_Str_images)
        d_optimizer.zero_grad()
        real_dresult = discriminator(c, real_Str_images)
        real_label = 0.1 * torch.rand(
            (batch_size, real_dresult.size(1), real_dresult.size(2), real_dresult.size(3))) + 0.9

        fake_dresult = discriminator(c, fake_Str_images)
        # print('fake_dresult',fake_dresult)
        fake_lable = 0.1 * torch.rand(
            (batch_size, fake_dresult.size(1), fake_dresult.size(2), fake_dresult.size(3))).to(device)
        real_label = real_label.to(device)

        real_loss = bce(real_dresult, real_label)
        fake_loss = bce(fake_dresult, fake_lable)
        # print('fake_loss', fake_loss)
        dloss = (real_loss + fake_loss) / 2

        dloss.backward(retain_graph=True)
        d_optimizer.step()
        scheduler_d.step(dloss)
        train_dloss.append(dloss.item())

        ###################训练G####################
        fake_Str_images = generator(c, z)
        g_optimizer.zero_grad()
        g_result = discriminator(c, fake_Str_images)
        ones = torch.ones(batch_size, g_result.size(1), g_result.size(2), g_result.size(3)).to(device)
        gloss = bce(g_result, ones)

        gloss.backward()
        g_optimizer.step()
        train_gloss.append(gloss.item())
        scheduler_g.step(gloss)

        ############处理生成图片
        def trainacc():
            train_tp = []
            train_tn = []
            train_fp = []
            train_fn = []
            for a in range(batch_size):
                save_image(denorm(real_Str_images[a]), os.path.join(real_img_path,
                                                                    'train_real_images-{}.png'.format(
                                                                        (a + 1))))
                save_image(denorm(fake_Str_images[a]), os.path.join(fake_img_path,
                                                                    'train_fake_images-{}.png'.format(
                                                                        (a + 1))))
                # print('fake_Str_images',fake_Str_images)
                # print('name[a]',name[a])
                fake_seq = Correction_unit(fake_img_path + '/train_fake_images-{}.png'.format((a + 1)), name[a],
                                           RNAlen[a], fake_Str_images.size(2))
                real_seq, len = Structure_matrix(real_img_path + '/train_real_images-{}.png'.format((a + 1)),
                                                 fake_img_path + '/train_fake_images-{}.png'.format((a + 1)),
                                                 RNAlen[a])
                # print(f"real_seq: {real_seq}")
                # print(f"fake_seq: {fake_seq}")

                TP, TN, FP, FN, acc, TPR, FPR = calculation(real_seq, fake_seq, len)
                train_tp.append(TP)
                train_tn.append(TN)
                train_fp.append(FP)
                train_fn.append(FN)
            return sum(train_tp), sum(train_tn), sum(train_fp), sum(train_fn), acc, TPR, FPR

        TP, TN, FP, FN, acc, TPR, FPR = trainacc()
    TP_total.append(TP)
    TN_total.append(TN)
    FP_total.append(FP)
    FN_total.append(FN)
    return np.mean(train_gloss), np.mean(train_dloss), sum(TP_total), sum(TN_total), sum(FP_total), sum(FN_total)


# 验证集
def Val_dataset(g_model, d_model, Structure_matrix, calculation):
    g_model.eval()
    d_model.eval()
    val_gloss = []
    val_dloss = []
    TP_total = []
    TN_total = []
    FP_total = []
    FN_total = []
    for i, data in enumerate(validate_loader):
        real_Seq_images, real_Str_images, RNAlen, name, rlabel = data
        real_Str_images = real_Str_images.to(device)
        c = real_Seq_images.to(device)
        z = torch.randn([batch_size, c.size(1), c.size(2), c.size(3)]).to(device)  # 噪声

        ##################################################
        fake_Str_images = generator(c, z)
        real_dresult = discriminator(c, real_Str_images)
        real_label = 0.1 * torch.rand(
            (batch_size, real_dresult.size(1), real_dresult.size(2), real_dresult.size(3))) + 0.9
        fake_dresult = discriminator(c, fake_Str_images)

        fake_lable = 0.1 * torch.rand(
            (batch_size, fake_dresult.size(1), fake_dresult.size(2), fake_dresult.size(3))).to(device)
        real_label = real_label.to(device)
        real_loss = bce(real_dresult, real_label)

        fake_loss = bce(fake_dresult, fake_lable)

        dloss = real_loss + fake_loss
        val_dloss.append(dloss.item())

        ########################################################
        fake_Str_images = generator(c, z)
        g_result = discriminator(c, fake_Str_images)
        ones = torch.ones(batch_size, g_result.size(1), g_result.size(2), g_result.size(3)).to(device)

        gloss = bce(g_result, ones)
        val_gloss.append(gloss.item())

        def valacc():
            val_tp = []
            val_tn = []
            val_fp = []
            val_fn = []
            for a in range(batch_size):
                save_image(denorm(real_Str_images[a]), os.path.join(real_img_path,
                                                                    'val_real_images-{}.png'.format(

                                                                        (a + 1))))
                save_image(denorm(fake_Str_images[a]), os.path.join(fake_img_path,
                                                                    'val_fake_images-{}.png'.format(

                                                                        (a + 1))))

                fake_seq = Correction_unit(fake_img_path + '/val_fake_images-{}.png'.format((a + 1)), name[a],
                                           RNAlen[a], fake_Str_images.size(2))
                real_seq, len = Structure_matrix(real_img_path + '/val_real_images-{}.png'.format((a + 1)),
                                                 fake_img_path + '/val_fake_images-{}.png'.format((a + 1)),
                                                 RNAlen[a])
                TP, TN, FP, FN, acc, TPR, FPR = calculation(real_seq, fake_seq, len)
                val_tp.append(TP)
                val_tn.append(TN)
                val_fp.append(FP)
                val_fn.append(FN)
            return sum(val_tp), sum(val_tn), sum(val_fp), sum(val_fn), acc, TPR, FPR

        TP, TN, FP, FN, acc, TPR, FPR = valacc()
        TP_total.append(TP)
        TN_total.append(TN)
        FP_total.append(FP)
        FN_total.append(FN)
    return np.mean(val_gloss), np.mean(val_dloss), sum(TP_total), sum(TN_total), sum(FP_total), sum(FN_total)


# 测试集
def Test_dataset(g_model, d_model, Structure_matrix, calculation, epoch):
    g_model.eval()
    d_model.eval()
    test_gloss = []
    test_dloss = []
    TP_total = []
    TN_total = []
    FP_total = []
    FN_total = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            real_Seq_images, real_Str_images, RNAlen, name, rlabel = data
            real_Str_images = real_Str_images.to(device)
            c = real_Seq_images.to(device)
            z = torch.randn([batch_size, c.size(1), c.size(2), c.size(3)]).to(device)  # 噪声

            ##################################################
            fake_Str_images = generator(c, z)
            real_dresult = discriminator(c, real_Str_images)
            real_label = 0.1 * torch.rand(
                (batch_size, real_dresult.size(1), real_dresult.size(2), real_dresult.size(3))) + 0.9
            fake_dresult = discriminator(c, fake_Str_images)

            fake_lable = 0.1 * torch.rand(
                (batch_size, fake_dresult.size(1), fake_dresult.size(2), fake_dresult.size(3))).to(device)
            real_label = real_label.to(device)
            real_loss = bce(real_dresult, real_label)

            fake_loss = bce(fake_dresult, fake_lable)

            dloss = real_loss + fake_loss
            test_dloss.append(dloss.item())

            ########################################################
            fake_Str_images = generator(c, z)
            g_result = discriminator(c, fake_Str_images)
            ones = torch.ones(batch_size, g_result.size(1), g_result.size(2), g_result.size(3)).to(device)

            gloss = bce(g_result, ones)
            test_gloss.append(gloss.item())

            def testacc():
                test_tp = []
                test_tn = []
                test_fp = []
                test_fn = []
                for a in range(batch_size):
                    save_image(denorm(real_Str_images[a]), os.path.join(real_img_path,
                                                                        'test_real_images-{}.png'.format(

                                                                            (a + 1))))
                    save_image(denorm(fake_Str_images[a]), os.path.join(fake_img_path,
                                                                        'test_fake_images-{}.png'.format(

                                                                            (a + 1))))

                    fake_seq = Correction_unit(fake_img_path + '/test_fake_images-{}.png'.format((a + 1)), name[a],
                                               RNAlen[a], fake_Str_images.size(2))
                    real_seq, len = Structure_matrix(real_img_path + '/test_real_images-{}.png'.format((a + 1)),
                                                     fake_img_path + '/test_fake_images-{}.png'.format((a + 1)),
                                                     RNAlen[a])
                    TP, TN, FP, FN, acc, TPR, FPR = calculation(real_seq, fake_seq, len)
                    test_tp.append(TP)
                    test_tn.append(TN)
                    test_fp.append(FP)
                    test_fn.append(FN)
                return sum(test_tp), sum(test_tn), sum(test_fp), sum(test_fn), acc, TPR, FPR

            TP, TN, FP, FN, acc, TPR, FPR = testacc()
            TP_total.append(TP)
            TN_total.append(TN)
            FP_total.append(FP)
            FN_total.append(FN)
        return np.mean(test_gloss), np.mean(test_dloss), sum(TP_total), sum(TN_total), sum(FP_total), sum(FN_total)


def model():
    epsilon = 1e-7
    for epoch in tqdm(range(total_epochs)):
        train_gloss, train_dloss, train_TP, train_TN, train_FP, train_FN = Train_dataset(generator, discriminator,
                                                                                         Structure_matrix, calculation)
        # val_gloss, val_dloss, val_TP, val_TN, val_FP, val_FN = Val_dataset(generator, discriminator, Structure_matrix,
        #                                                                    calculation)

        # generator_path = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/pkl/generator{epoch}.pkl'.format(epoch=epoch + 1)
        # discriminator_path = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/pkl/discriminator{epoch}.pkl'.format(epoch=epoch + 1)
        #
        # 生成目录路径（去除文件名）
        generator_dir = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/pkl/'
        discriminator_dir = '/home/chenjingjing/Models/MSFF-CDCGAN/result/' + path + '/pkl/'

        # 检查并创建目录
        if not os.path.exists(generator_dir):
            os.makedirs(generator_dir)
        if not os.path.exists(discriminator_dir):
            os.makedirs(discriminator_dir)

        # 生成文件路径
        generator_path = generator_dir + 'generator{epoch}.pkl'.format(epoch=epoch + 1)
        discriminator_path = discriminator_dir + 'discriminator{epoch}.pkl'.format(epoch=epoch + 1)

        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)
        generator.load_state_dict(torch.load(generator_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
        train_acc = (train_TP + train_TN) / (train_TP + train_TN + train_FP + train_FN + epsilon)
        # test_c = []
        # test_c1 = np.zeros(6)
        # for i in range(10):
        test_gloss, test_dloss, test_TP, test_TN, test_FP, test_FN = Test_dataset(generator, discriminator,
                                                                                  Structure_matrix, calculation, epoch)


        test_acc = (test_TP + test_TN) / (test_TP + test_TN + test_FP + test_FN + epsilon)
        R = test_TP / (test_TP + test_FN + epsilon)
        P = test_TP / (test_TP + test_FP + epsilon)
        F1 = (2 * R * P) / (R + P + epsilon)
        # 计算 MCC
        numerator = (test_TP * test_TN) - (test_FP * test_FN)
        denominator = ((test_TP + test_FP) * (test_TP + test_FN) * (test_TN + test_FP) * (test_TN + test_FN)) ** 0.5
        test_mcc = numerator / (denominator + epsilon)
            # test_c.append([test_gloss, test_dloss,test_acc,R,P,F1])
        # for i ,j in enumerate(test_c):
        #     test_c1 = test_c1+np.array(test_c[i])
        # test_gloss = test_c1[0]/10
        # test_dloss = test_c1[1]/10
        # test_acc = test_c1[2]/10
        # R = test_c1[3]/10
        # P = test_c1[4]/10
        # F1 = test_c1[5]/10
        Train_gloss.write(str(train_gloss.item()) + '\n')
        Train_dloss.write(str(train_dloss.item()) + '\n')
        Train_acc.write(str(train_acc) + '\n')
        Test_gloss.write(str(test_gloss.item()) + '\n')
        Test_dloss.write(str(test_dloss.item()) + '\n')
        Test_acc.write(str(test_acc) + '\n')

        print("[Epoch %d/%d]" % (epoch + 1, total_epochs),
              'train_gloss:', train_gloss,
              'train_dloss:', train_dloss,
              'test_gloss:', test_gloss,
              'test_dloss:', test_dloss,
              'train_acc:', train_acc,
              'test_acc:', test_acc,
              'test_r:', R,
              'test_p:', P,
              'test_f1', F1,
              'test_mcc', test_mcc
              )
        if R>=0.90 and P >=0.90 :
            print("[This Epoch %d good]" % (epoch + 1))


# visdom(epoch,train_gloss,train_dloss,val_gloss,val_dloss,train_accuracy,val_accuracy)




def main():
    model()


if __name__ == '__main__':
    main()
