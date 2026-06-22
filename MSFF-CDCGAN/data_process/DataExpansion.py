# -*- coding: UTF-8 -*-
from PIL import Image
import os
from tqdm import tqdm  # Import tqdm for progress bars
# 数据扩充
def enlarge(path):
    # files = os.listdir(path)
    # for file in files:
    #     fname = path + file
    files = os.listdir(path)
    for file in tqdm(files, desc=f'Processing images in {path}', unit='file'):
        fname = os.path.join(path, file)

        im = Image.open(fname)
        out_5 = im.transpose(Image.FLIP_LEFT_RIGHT) #左右翻转
        out_6 = im.transpose(Image.FLIP_TOP_BOTTOM) #上下翻转

        img_name = fname.split("/")[-1]
        img_name = img_name.rstrip('.png')
        out_5.save(path+img_name+'-l-r.png')
        out_6.save(path+img_name+'-u-d.png')

def main():
    name = '5srna'
    path ='/home/chenjingjing/Models/MSFF-CDCGAN/picture/bprna_new/test/str_photo/'
    path1 = '/home/chenjingjing/Models/MSFF-CDCGAN/picture/bprna_new/test/seq_photo/'
    # path = r"../picture/"+name+"/str_photo/"
    # path1 = r"../picture/"+name+"/seq_photo/"
    enlarge(path)
    enlarge(path1)

if __name__ == '__main__':
    main()







