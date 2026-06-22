import torch
print(torch.__version__)          # PyTorch version
print(torch.version.cuda)          # CUDA version PyTorch was built with
print(torch.cuda.is_available())   # Should output "True"
from torch.backends import cudnn
print(cudnn.is_available())        # Should output "True"
print(cudnn.is_acceptable(torch.randn(1).cuda()))  # Should output "True"
print(torch.backends.cudnn.version())

import torch
x = torch.randn(3, 3).cuda()
print(x)  # Should show device='cuda:0'