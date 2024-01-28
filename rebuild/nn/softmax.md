`class torch.nn.Softmax(dim=None)`

Applies the Softmax function to an n-dimensional input Tensor **rescaling** them so that the elements of the n-dimensional output Tensor lie in the range `[0,1]` and sum to 1.

Softmax is defined as:

$$\text{Softmax}(x_i)=\dfrac{\exp(x_i)}{\sum_j\exp(x_j)}$$

– `dim` (int) A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

- return
  
    返回的shape同输入的shape一样。

例子：
```python
>>> m = nn.Softmax(dim=1)
>>> input = torch.tensor([[4, 1, 1, 4], [5, 2, 3, 0]]).float()
>>> m(input)
tensor([[0.4763, 0.0237, 0.0237, 0.4763],
        [0.8390, 0.0418, 0.1135, 0.0057]])

# dim 的意思
>>> m(input).sum(dim=1)
tensor([1.0000, 1.0000])
```
常用的意义：`dim=-1`，这里每行的和都为1，也就是说，每个样本的特征、类别 softmax的概率.
```python
>>> m = nn.Softmax(dim=-1)
>>> input = torch.randn(2, 3, 4)
>>> m(input)
tensor([[[0.6758, 0.2132, 0.0317, 0.0793],
         [0.0213, 0.6675, 0.1319, 0.1794],
         [0.3582, 0.0202, 0.2315, 0.3902]],

        [[0.5270, 0.2125, 0.2315, 0.0290],
         [0.5760, 0.0857, 0.1807, 0.1575],
         [0.0897, 0.0833, 0.5022, 0.3248]]])
```