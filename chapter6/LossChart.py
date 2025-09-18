import torch
from torch.utils.tensorboard import SummaryWriter

# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成数据
inputs = torch.rand(100, 3)  # 生成shape为(100,3)的tensor，里边每个元素的值都是0-1之间
weights = torch.tensor([[1.1], [2.2], [3.3]])
bias = torch.tensor(4.4)
targets = inputs @ weights + bias + 0.1 * torch.randn(100, 1)  # 增加一些随机，模拟真实情况

# 创建一个SummaryWriter实例
writer = SummaryWriter(log_dir="./lr/runs/")

# 初始化参数时直接放在CUDA上，并启用梯度追踪
w = torch.rand(3, 1, requires_grad=True, device=device)
b = torch.rand(1, requires_grad=True, device=device)

# 将数据移至相同设备
inputs = inputs.to(device)
targets = targets.to(device)

epoch = 10000
lr = 0.003

# 开始训练循环
for i in range(epoch):
    # 前向传播：计算模型输出
    outputs = inputs @ w + b  # 矩阵乘法：输入乘以权重加偏置
    
    # 计算损失函数（均方误差）
    loss = torch.mean(torch.square(outputs - targets))  # MSE损失
    
    # 打印当前损失值（每次迭代都打印）
    print("loss:", loss.item())  # .item()将tensor转为Python标量
    
    # 记录损失到TensorBoard，用于可视化训练过程
    # 参数：标签名称，损失值，当前步数
    writer.add_scalar("loss/train", loss.item(), i)
    
    # 反向传播：计算梯度
    loss.backward()  # 自动计算所有需要梯度的参数的梯度
    
    # 手动更新参数（梯度下降）
    with torch.no_grad():  # 禁用梯度追踪，避免更新操作被记录到计算图中
        w -= lr * w.grad  # 权重更新：w = w - 学习率 * 梯度
        b -= lr * b.grad  # 偏置更新：b = b - 学习率 * 梯度
    
    # 清零梯度，为下一次迭代做准备
    w.grad.zero_()  # 将权重梯度重置为0
    b.grad.zero_()  # 将偏置梯度重置为0

# 训练完成，输出最终的模型参数
print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)
