- [1. 手搓](#1-手搓)
- [2. nn.Linear](#2-nnlinear)
- [3. 多维度随便乘](#3-多维度随便乘)

---
## 1. 手搓

在数学表示上区分行列向量，在程序中则不用区别。

> 单个向量

$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b$

$\hat{y}_i = w_1  x_{i,1} + ... + w_d  x_{i,d} + b$

$\hat{y}_i = <\mathbf{x}_i, \mathbf{w}> + b = x^\top w +b = w^\top x + b$, 这里$\mathbf{x}_i$ 的shape是`[d]`, $\mathbf{w}$ 的shape是`[d]`，用向量内积。

!!! note 转置？ 
    
    我们总是搞混一点，以为这里的$\top$是表示转置！但其实不是，有没有$\top$只是为标识向量是行向量还是列向量。

    所以，向量表示和矩阵表示都是统一的 $y=xw+b$

> 矩阵

${\hat{\mathbf{y}}} = \mathbf{x} \mathbf{w} + \mathbf{b}$, 这里$\mathbf{x}$ 的shape是`[batch, d_input]` (**每一行是一个样本, 每一列是一种特征**), $\mathbf{w}$ 的shape是`[d_input, d_output]`, $\mathbf{b}$ 是`[batch, d_output]`. 而且其一可以退化为向量。

```python
# 向量-向量
>>> x_i = torch.randn(4) 
>>> w = torch.randn(4) 
>>> x_i @ w
tensor(-0.5885)

# 矩阵-矩阵
>>> x = torch.randn(2, 4)
>>> w = torch.randn(4, 3)
>>> x @ w
tensor([[-2.5128,  0.4920, -1.8654],
        [-0.0368,  0.5289,  1.2328]])

# 矩阵-向量
>>> x = torch.randn(2, 4)
>>> w = torch.randn(4)
>>> x @ w
tensor([ 1.1529, -0.9762])

# 向量-矩阵
>>> x = torch.randn(4)
>>> w = torch.randn(4, 3)
>>> x @ w
tensor([-0.8961, -4.8806, -3.0002])
```

## 2. nn.Linear

`nn.Linear`迷惑人的地方在于其weight是(3,4)。但其相乘的时候是`xw^T+b`。

所以如果weight如手搓的(4,3)，那不还是`xw+b`.

输入`[2,4]`,权重`4,3`，那么结果就是`2,3`，这样多么直接。而`nn.Linear`这种表示反直觉，多此一举。


```python
>>> nn.Linear(4, 3)
Linear(in_features=4, out_features=3, bias=True)
>>> nn.Linear(4, 3).weight
Parameter containing:
tensor([[-0.3712, -0.3247,  0.0623, -0.0977],
        [ 0.1350, -0.1824, -0.0635,  0.4179],
        [ 0.1864, -0.0315, -0.3408, -0.2920]], requires_grad=True)
>>> nn.Linear(4, 3).weight.shape    # 注意不同
torch.Size([3, 4])
```

## 3. 多维度随便乘

```python
>>> nn.Linear(4,3)(torch.randn(8, 7, 6, 5, 4)).shape
torch.Size([8, 7, 6, 5, 3])

>>> x = torch.randn(8, 7, 6, 5 ,4)
>>> w = torch.randn(8, 7, 6, 4 ,3)
>>> (x @ w).shape
torch.Size([8, 7, 6, 5, 3])
```