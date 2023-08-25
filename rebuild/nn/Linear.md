```python
>>> nn.Linear(4, 3)
Linear(in_features=4, out_features=3, bias=True)
>>> nn.Linear(4, 3).weight
Parameter containing:
tensor([[-0.3712, -0.3247,  0.0623, -0.0977],
        [ 0.1350, -0.1824, -0.0635,  0.4179],
        [ 0.1864, -0.0315, -0.3408, -0.2920]], requires_grad=True)
>>> nn.Linear(4, 3).weight.shape    # 所以 w 要转置
torch.Size([3, 4])
```

$f(x)=x * w^{T} + b$