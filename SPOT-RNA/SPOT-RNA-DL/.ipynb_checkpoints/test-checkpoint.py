import torch
print(torch.__version__)  # 应显示类似1.12.0+cu113
print(torch.cuda.is_available())  # 应返回True