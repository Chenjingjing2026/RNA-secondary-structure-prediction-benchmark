import numpy as np

# 加载 .npz 文件
npzfile = np.load('/home/chenjingjing/Models/Neuralfold/result/NEURALfold_params.data0')

# 输出文件内的所有键
print(npzfile.files)