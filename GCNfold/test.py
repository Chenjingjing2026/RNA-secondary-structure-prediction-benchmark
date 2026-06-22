import torch
import dgl

print("PyTorch CUDA Version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("DGL Version:", dgl.__version__)
