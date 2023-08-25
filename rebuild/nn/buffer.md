```python
register_buffer(name, tensor, persistent=True)
```

- `tensor`:

    传入Tensor即可，不需要将其转成gpu。当网络进行`.cuda`时候，会自动将里面的层的参数，buffer等转换成相应的GPU上。

    If `None`, then operations that run on buffers, such as `cuda`, are ignored. 
    
    If `None`, the buffer is not included in the module’s `state_dict`.

- `persistent`:

    网络存储时也会将buffer存下，当网络load模型时，会将存储的模型的buffer也进行赋值。
  
    non-persistent buffer is not included in the module’s `state_dict`.


访问：Buffers can be accessed as attributes using given `name`.
```python
class xxx:
    def __init__(self):
        self.register_buffer('running_mean', torch.zeros(num_features))
    
    def forward(self):
        self.running_mean
```

`nn.Parameter` 在每次 `optim.step()` 会得到更新，而`register_buffer` 只在 `forward()` 中进行更新。