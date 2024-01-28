```python
self.blocks = nn.ModuleList([nn.ModuleList([nn.Linear(W, W) for i in range(4)]) for i in range(4) ])


# forward
h = self.input_net(usvt_embed)
for block in self.blocks:
    residual = h
    for liner in block:
        h = liner(h)
        h = F.relu(h)
    h = h + residual
    h = F.relu(h)
```