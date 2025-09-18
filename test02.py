"""
线性回归训练脚本
本代码实现了一个简单的线性回归模型，使用PyTorch进行训练，并结合SwanLab记录实验数据。
主要功能包括：
1. 数据生成与标准化：生成特征范围差异较大的数据，并进行标准化处理。
2. 模型训练：使用均方误差 (MSE) 作为损失函数，通过梯度下降法优化权重和偏置。
3. 实验记录：通过SwanLab记录超参数、损失值和梯度范数，便于后续分析与可视化。
4. 可视化结果：绘制真实值与预测值的对比图。
依赖：torch, swanlab, matplotlib
注意：代码支持CUDA加速，需确保运行环境支持CUDA。

"""

import torch
from torch.utils.tensorboard import SummaryWriter
import swanlab
import time
import matplotlib.pyplot as plt


# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成数据 (特征范围差距过大，会导致梯度nan)========================================================
inputs = torch.cat([
    torch.rand(100, 1),                # 第一列：0-1
    torch.rand(100, 1) * 20 - 10,      # 第二列：-10到10
    torch.rand(100, 1) * 900 + 100     # 第三列：100到1000
], dim=1)

weights = torch.tensor([[1.1], [2.2], [3.3]])
bias = torch.tensor(4.4)
targets = inputs @ weights + bias + 0.1 * torch.randn(100, 1)  # 增加一些随机，模拟真实情况

#计算特征的均值和标准差
mean = inputs.mean(dim=0)
std = inputs.std(dim=0)
#对特征进行标准化
inputs_norm = (inputs-mean)/std

# 创建一个SummaryWriter实例
# writer = SummaryWriter(log_dir="C:/projects/lr/runs/")

# 初始化参数时直接放在CUDA上，并启用梯度追踪
w = torch.randn(3, 1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)

# 将数据移至相同设备
inputs_norm = inputs_norm.to(device)
targets = targets.to(device)

# 超参数========================================================
epoch = 10000
lr = 0.001

# 初始化SwanLab
run = swanlab.init(
    # 设置项目
    project="RethinkFunDeepLearning",
    experiment_name="linear_regression"+ time.strftime("%Y-%m-%d %H:%M:%S"),
    description="线性回归实验" ,

    # 跟踪超参数与实验元数据
    config={
        "learning_rate": lr,
        "epochs": epoch,
    },
)

# 训练循环========================================================
for i in range(epoch):
    # 前向传播：计算预测值
    outputs = inputs_norm @ w + b  # 矩阵乘法计算预测值

    # 计算损失函数，这里使用均方误差 (MSE)
    loss = torch.mean(torch.square(outputs - targets))  # 预测值与真实值的平方差的均值
    print("loss:", loss.item())  # 打印当前损失值

    # 记录损失到SwanLab，方便后续可视化与分析
    # 参数分别为：当前epoch，学习率，损失值
    swanlab.log({"epoch": i, "learning_rate": lr, "loss": loss})

    # 反向传播：计算梯度
    loss.backward()  # 自动计算损失函数对参数 w 和 b 的梯度
    # print("w.grad:", w.grad)  # 打印权重 w 的梯度
    # print("b.grad:", b.grad)  # 打印偏置 b 的梯度

    # 使用 torch.no_grad() 表示以下操作不需要计算梯度
    with torch.no_grad():
        # 更新权重和偏置，使用梯度下降法
        w -= lr * w.grad  # 更新权重 w
        b -= lr * b.grad  # 更新偏置 b

    # 记录梯度的范数到SwanLab，方便观察梯度变化
    swanlab.log({"w_grad": w.grad.norm(), "b_grad": b.grad.norm()})

    # 清零梯度，避免梯度累积
    w.grad.zero_()  # 清零权重 w 的梯度
    b.grad.zero_()  # 清零偏置 b 的梯度

# 打印训练后的参数
print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)

# 可视化结果========================================================
# 使用训练后的权重和偏置计算预测值
with torch.no_grad():
    predicted = inputs_norm @ w + b

# 将数据从GPU移回CPU以便绘图
inputs_cpu = inputs.cpu()
targets_cpu = targets.cpu()
predicted_cpu = predicted.cpu()

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(targets_cpu)), targets_cpu, label="True Values", color="blue", alpha=0.6)
plt.scatter(range(len(predicted_cpu)), predicted_cpu, label="Predicted Values", color="red", alpha=0.6)
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("True vs Predicted Values")
plt.legend()
plt.show()
