```python
torch.tensor([-2., -1., 0., 255., 255.5, 256, 257.]).to(torch.uint8)
tensor([254, 255,   0, 255, 255,   0,   1], dtype=torch.uint8)
```

超过[0, 255]的都会回转，引起错误。