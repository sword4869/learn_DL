```python
import torch
torch.autograd.set_detect_anomaly(True)
```
this is a debugging option, 用于开启自动求导过程中的异常检测功能 `nan`。PyTorch 会输出相关的错误信息以及出错的代码位置。

Slow down the training process by a large amount (almost 50% in my computer). I'd like to suggest to **turn it off as default in the release code** .

异常检测功能只对使用了反向传播的自动求导操作生效，对使用 `torch.no_grad()` 上下文管理器的自动求导操作不会产生影响。