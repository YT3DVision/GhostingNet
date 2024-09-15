import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
print(torch.cuda.is_available())
print(CUDA_HOME)
