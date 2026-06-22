# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
# """
# Created on Wed Jan 17 11:25:12 2018
#
# @author: zhangch
# """
#
# from __future__ import division
# import os
# from PIL import Image
# import scipy.misc
#
# source_path = '/home/chenjingjing/Models/CDPfold/data/RNAStrAlign/'
#
# aim_path = '/home/chenjingjing/Models/CDPfold/data/RNAStrAlign//train_set/'
#
# if __name__ == "__main__":
#     pathDir = os.listdir(source_path)
#     for i in pathDir:
#         #print source_path+i
#         image = Image.open(source_path+i)
#         image = image.resize((128,19))
#         scipy.misc.imsave(aim_path+i, image)

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:25:12 2018

@author: zhangch
"""

from __future__ import division
import os
from PIL import Image  # 已经导入PIL
from tqdm import tqdm  # 导入 tqdm

# source_path = '/home/zhangch/RSS/tRNA/train_labelpng2/'
# aim_path = '/home/zhangch/RSS/train_set/'

# source_path = '/home/chenjingjing/Models/CDPfold/data/RNAStrAlign/png2/'
source_path = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/test_labelpng/'
aim_path = '/home/chenjingjing/Models/CDPfold/data/bprna_new/test/test_set/'
if not os.path.exists(aim_path):
    os.makedirs(aim_path)


if __name__ == "__main__":
    pathDir = os.listdir(source_path)
    # for i in pathDir:
    # 使用 tqdm 显示进度条
    for i in tqdm(pathDir, desc="Processing images"):
        # print source_path+i
        image = Image.open(source_path + i)
        image = image.resize((128, 19))
        # print(aim_path + i)
        # 使用Pillow的save方法替代scipy.misc.imsave
        image.save(aim_path + i)  # 保存图像
