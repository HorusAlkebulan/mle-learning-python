import torch


print(f"Torch version: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")
