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
X.requires_grad_()    # X.requires_grad_(True), default is True
```
```python
>>> X = torch.randint(2, (2, 3)).float()
>>> X.requires_grad = False
>>> X
tensor([[0., 0., 1.],
        [1., 0., 0.]])


>>> X = torch.randint(2, (2, 3)).float()
>>> X.requires_grad_(False)
tensor([[0., 0., 0.],
        [1., 0., 1.]])
```

To freeze parts of your **model**, simply apply `.requires_grad_(False)` to the parameters that you don’t want updated.


> 与 `no_grad()`: `no_grad()` 临时覆盖 `require_grad=True`: 

[Computations in no-grad mode behave as if none of the inputs require grad. In other words, computations in no-grad mode are never recorded in the backward graph **even if there are inputs that have `require_grad=True`**.](https://pytorch.org/docs/stable/notes/autograd.html#)


### 1.1.2. 计算梯度

`xxxxx.backward()` 计算谁的梯度写谁



```python
>>> X = torch.randint(2, (2, 3)).float()
>>> X.requires_grad_()
tensor([[1., 1., 0.],
        [0., 0., 1.]], requires_grad=True)
>>> y = X * X
>>> y.sum().backward()

```

`y.sum().backward()`: 当`xxxxx` 是个N-D时，化成0-D的标量。因为不是标量，越求导维度越大。这就是为什么 `loss.backward()`, loss is a fuction returning a **scalar**.

```python
# X的数据本身
>>> X.data
tensor([[1., 1., 0.],
        [0., 0., 1.]])

# 获取得到的梯度
>>> X.grad
tensor([[2., 2., 0.],
        [0., 0., 2.]])

# 手动梯度清零：在默认情况下，PyTorch会累积梯度
>>> X.grad.zero_()
tensor([[0., 0., 0.],
        [0., 0., 0.]])

# 清零和不需要梯度不一样，后者是None
>>> X.requires_grad_(False)
tensor([[1., 1., 0.],
        [0., 1., 0.]])
>>> X.grad is None
True
```

## 1.2. detach
[clone, detach](https://zhuanlan.zhihu.com/p/389738863): 
- `clone()`: a和b不是同一个对象, 也不指向同一块内存地址, 但b的requires_grad属性和a的一样，同样是True or False.
  
  返回的tensor是中间节点，梯度会流向原tensor，即返回的tensor的梯度会叠加在原tensor上

- `detach()`: 从计算图中脱离出来。a和b不是同一个对象了, 但指向同一块内存地址, 不再需要梯度，即`requires_grad=False`

- `tensor.clone().detach()` 还是 `tensor.detach().clone()` 都行。

  修改其中一个tensor的值，另一个也会改变

[detach, detach_](https://blog.csdn.net/qq_27825451/article/details/95498211)

[在模型中时开时关detach()而不能阻止权重更新](../%E6%95%B0%E6%8D%AE%E9%9B%86/unsolvable/net_recurring_detach.ipynb)
