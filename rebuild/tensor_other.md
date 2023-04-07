torch.linspace: <https://zhuanlan.zhihu.com/p/114663117>

- `torch.linspace(start, end, steps=100, out=None)`, [start, ..., end], 包括头尾， 共step个元素。 

torch.meshgrid: <https://blog.csdn.net/weixin_39504171/article/details/106356977>

- `x, y = torch.meshgrid(a, b)`， x和y的shape是`[len(a), len(b)]`，`x`是复制列向量`a`，`y`是复制行向量`b`

```python
# 棋盘坐标，[10, 10, 2]
coords = torch.stack(
    torch.meshgrid(
        torch.linspace(0, 9, 10), 
        torch.linspace(0, 9, 10),
    ), -1)

[[[0., 0.], [0., 1.], [0., 2.], ..., [0., 9.]],
 [[1., 0.], [1., 1.], [1., 2.], ..., [1., 3.],
 ...,
 [[9., 0.], [9., 1.], [9., 2.], ..., [9., 9.]]]
```