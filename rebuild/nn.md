## ModuleList

[详解PyTorch中的ModuleList和Sequential](https://zhuanlan.zhihu.com/p/75206669)


网络中有很多相似或者重复的层
```python
def __init__(self):
    self.views_linears = nn.ModuleList(
        [nn.Linear(W + W, W)] +
        [nn.Linear(W, W) for i in range(4)]
    )

def forward(self, inputs):
    h = inputs
    for linear in self.views_linears:
        h = linear(h)
        h = F.relu(h)
```

> `nn.ModuleList([List])`不像`nn.Sequential()`能直接处理输入，必须用for循环列表的元素来一个个处理输入。

```python
# NotImplementedError: Module [ModuleList] is missing the required "forward" function
self.views_linears = nn.ModuleList(
    [nn.Linear(W + W, W), nn.ReLU()]
)
h = self.blocks(inputs)
```

> 不能用列表，只有用`nn.ModuleList([List])`包裹起来的列表才会被注册到整个网络中。
```python
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:1! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)
self.views_linears = [nn.Linear(W + W, W)] + [nn.Linear(W, W) for i in range(4)]
```