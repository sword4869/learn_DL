- [1. grad](#1-grad)
  - [1.1. grad of tensor](#11-grad-of-tensor)
  - [Net](#net)

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
```


detach:
[clone, detach](https://zhuanlan.zhihu.com/p/389738863): 
- `clone()`: a和b不是同一个对象, 也不指向同一块内存地址, 但b的requires_grad属性和a的一样，同样是True or False.
  
  返回的tensor是中间节点，梯度会流向原tensor，即返回的tensor的梯度会叠加在原tensor上

- `detach()`: 从计算图中脱离出来。a和b不是同一个对象了, 但指向同一块内存地址, 但不涉及梯度计算，即requires_grad=False

- `tensor.clone().detach()` 还是 `tensor.detach().clone()` 都行。

  修改其中一个tensor的值，另一个也会改变

[detach, detach_](https://blog.csdn.net/qq_27825451/article/details/95498211)

[在模型中时开时关detach()而不能阻止权重更新](../unsolvable/net_recurring_detach.ipynb)

## Net


不加载梯度

![图 7](../images/87519e852836e4157f551b99c9be7374a8c0ad88f64b40b2c727bd90a0b4d521.png)  

训练、验证、测试:

这里`optimizer.zero_grad()`写在for循环开始、写在`optimizer.step()`后面的都行。

![图 5](../images/796ec7e3493ded28ac0da0a00899df2bd30196b42b7b9a7d4351055ff2656656.png)  
 



![图 4](../images/77bdacd36dfa57e1f72fe6bc5641e2113c9104ec83594fe69cf14081d8c2bea8.png)  


![图 6](../images/9c5c01f071d2528b0b0b415fad698050d815069f68811e78b415d5f4ae393816.png)  


