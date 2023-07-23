- [1. grad](#1-grad)
  - [1.1. grad of tensor](#11-grad-of-tensor)
    - [1.1.1. 创建梯度](#111-创建梯度)
    - [1.1.2. 计算梯度](#112-计算梯度)
  - [1.2. detach](#12-detach)

# 1. grad
## 1.1. grad of tensor

### 1.1.1. 创建梯度

only Tensors of **floating** point and complex dtype can require gradients.
```python
# (1) 默认创建的tensor没有grad
torch.arange(4.0, requires_grad=True)

# (2) requires_grad = True
X = torch.randint(2, (2, 3)).float()
X.requires_grad = True
# tensor([[0., 1., 0.],
#         [0., 0., 1.]], requires_grad=True)

# (3) requires_grad_(True)
X = torch.randint(2, (2, 3)).float()
X.requires_grad_(True)
```

### 1.1.2. 计算梯度
```python
y = X * X

# 计算梯度: 计算谁的梯度写谁
y.backward()

# 当y是个N-D时，化成0-D的标量。因为不是标量，越求导维度越大。
# 这就是为什么 loss.backward(), loss is a fuction returning a scalar.
y.sum().backward()

# 获取得到的梯度
X.grad


# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
X.grad.zero_()
```


## 1.2. detach
[clone, detach](https://zhuanlan.zhihu.com/p/389738863): 
- `clone()`: a和b不是同一个对象, 也不指向同一块内存地址, 但b的requires_grad属性和a的一样，同样是True or False.
  
  返回的tensor是中间节点，梯度会流向原tensor，即返回的tensor的梯度会叠加在原tensor上

- `detach()`: 从计算图中脱离出来。a和b不是同一个对象了, 但指向同一块内存地址, 但不涉及梯度计算，即requires_grad=False

- `tensor.clone().detach()` 还是 `tensor.detach().clone()` 都行。

  修改其中一个tensor的值，另一个也会改变

[detach, detach_](https://blog.csdn.net/qq_27825451/article/details/95498211)

[在模型中时开时关detach()而不能阻止权重更新](../unsolvable/net_recurring_detach.ipynb)
