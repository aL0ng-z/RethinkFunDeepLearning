"""
PyTorch基础操作和性能测试脚本
演示PyTorch张量操作、维度扩展和CPU/GPU矩阵乘法性能对比。
依赖：torch, numpy, time
注意：GPU测试需CUDA环境。
"""

import torch
import numpy as np
import time

arr = np.array([[5, 3], [1, 9]])
print(arr)

# 测试张量
x = torch.randn(10, 2, 6)
print(x[1,1,:])
print(x.size())
print(x.shape)
print(x.dim())
print(x.numel())
print(x.dtype)
print(x.device)

# 测试维度扩展
temp = torch.randn(2, 3)
temp1 = torch.unsqueeze(temp,0)
temp2 = temp.unsqueeze(1)
print(temp)
print(temp1,temp1.shape,"\n",temp1[0,:,:])
print(temp2,temp2.shape,"\n",temp2[0,:,:])


# 计算性能测试
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

size = 50000  # 矩阵大小
A_cpu = torch.randn(size, size) # 默认在CPU上创建tensor
B_cpu = torch.randn(size, size)

# 在 CPU 上计算
start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)  # 矩阵乘法
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# 在 GPU 上计算
A_gpu = A_cpu.to(device) # 将tensor转移到GPU上
B_gpu = B_cpu.to(device)

start_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.cuda.synchronize()  # 确保GPU计算完成
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"CPU time: {cpu_time:.6f} sec")
if torch.cuda.is_available():
    print(f"GPU time: {gpu_time:.6f} sec")
else:
    print("GPU not available, skipping GPU test.")