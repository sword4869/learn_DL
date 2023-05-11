[详解PyTorch中的ModuleList和Sequential](https://zhuanlan.zhihu.com/p/75206669)


有的时候网络中有很多相似或者重复的层，我们一般会考虑用 for 循环来创建它们，而不是用`nn.Sequential()`一行一行地写，比如：
```python
def __init__(self):
    # 残差连接
    self.views_linears = nn.ModuleList(
        [nn.Linear(self.input_dim, W)] +
        [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_dim, W) for i in range(D-1)] +
        [nn.Linear(W, W//2)]
    )

def forward(self, inputs):
    h = inputs
    for i, l in enumerate(self.views_linears):
        h = self.views_linears[i](h)
        h = F.relu(h)
        if i in self.skips:
            h = torch.cat([h, inputs], -1)
```