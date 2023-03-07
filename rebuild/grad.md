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

![图 7](../images/87519e852836e4157f551b99c9be7374a8c0ad88f64b40b2c727bd90a0b4d521.png)  

训练、验证、测试

![图 5](../images/796ec7e3493ded28ac0da0a00899df2bd30196b42b7b9a7d4351055ff2656656.png)  
 


For every **batch**( not epoch ) of data:
1. Call `optimizer.zero_grad()` to reset gradients of model parameters.
2. Call `loss.backward()` to backpropagate gradients of prediction loss.
3. Call `optimizer.step()` to adjust model parameters.


![图 4](../images/77bdacd36dfa57e1f72fe6bc5641e2113c9104ec83594fe69cf14081d8c2bea8.png)  


![图 6](../images/9c5c01f071d2528b0b0b415fad698050d815069f68811e78b415d5f4ae393816.png)  
