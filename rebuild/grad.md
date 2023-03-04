- [1. grad](#1-grad)
  - [1.1. grad of tensor](#11-grad-of-tensor)

# 1. grad
## 1.1. grad of tensor

创建梯度
```python
# (1) 默认创建的tensor没有grad
torch.arange(4.0, requires_grad=True)

# (2)
# only Tensors of floating point and complex dtype can require gradients
X = torch.randint(2, (2, 3)).float()
X.requires_grad = True
# tensor([[0., 1., 0.],
#         [0., 0., 1.]], requires_grad=True)

# (3)
X = torch.randint(2, (2, 3)).float()
X.requires_grad_(True)
```

计算梯度
```python
y = X * X

# 计算梯度: 计算谁的梯度写谁
y.backward()

# 当y是个N-D时，化成0-D的标量。因为不是标量，越求导维度越大。
y.sum().backward()

# 获取得到的梯度
X.grad


# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
X.grad.zero_()

# 当成常数
u = y.detach()
```



不加载梯度
```python
# 不需要计算梯度时，比如评估loss时（不需要反向传播来更新，这不是在训练）
with torch.no_grad():
    ...
```
```python
# 评估模式，也不加载梯度
net.eval()
```