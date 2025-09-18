import torch

x = torch.tensor(1.0, requires_grad=True) #指定需要计算梯度
y = torch.tensor(1.0, requires_grad=True) #指定需要计算梯度
v = 3*x+4*y
u = torch.square(v)
z = torch.log(u)

z.backward() #反向传播求梯度

print("x grad:", x.grad)
print("y grad:", y.grad)