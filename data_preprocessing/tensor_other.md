## torch.linspace

<https://zhuanlan.zhihu.com/p/114663117>

- `torch.linspace(start, end, steps=100, out=None)`, [start, ..., end], 包括头尾， 共step个元素。 





## einsum einops

### torch.einsum

> 矩阵的乘积

假设矩阵 $A \in \mathbb{R}^{I \times K}$ 矩阵 $B \in \mathbb{R}^{K \times J}$ ，两个矩阵的乘积 $C$ 的维度可以表示为 $\mathbb{R}^{I \times J}$

用爱因斯坦求和约定可以如下表示：

$$
C_{ij} = (AB)_{ij}= \sum_{k=1}^{N}A_{ik} B_{kj} \\
$$

```python
>>> a = torch.arange(4).reshape(2,2)
>>> b = torch.arange(4,8).reshape(2,2)
>>> a
tensor([[0, 1],
        [2, 3]])
>>> b
tensor([[4, 5],
        [6, 7]])
>>> torch.einsum('ij,jk->ik', a, b)
tensor([[ 6,  7],
        [26, 31]])
>>> a @ b
tensor([[ 6,  7],
        [26, 31]])
```

$A B^\top$

```python
>>> torch.einsum('ik,jk->ij', a, b)
tensor([[ 5,  7],
        [23, 33]])
>>> a @ b.t()
tensor([[ 5,  7],
        [23, 33]])
```

$A^\top B$
```python
>>> torch.einsum('ki,kj->ij', a, b)
tensor([[12, 14],
        [22, 26]])
>>> a.t() @ b
tensor([[12, 14],
        [22, 26]])
```
> 求和

```python
>>> torch.einsum('ij->', a)
tensor(6)

# 行求和
>>> torch.einsum('ij->i', a)
tensor([1, 5])

# 列求和
>>> torch.einsum('ij->j', a)
tensor([2, 4])
```

> 转置

```python
>>> torch.einsum('ij->ji', a)
tensor([[0, 2],
        [1, 3]])
```