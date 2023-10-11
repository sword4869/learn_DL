`torchvision.utils.save_image`的内部代码
```python
# grid.mul(255) 负责将[0,1]到[0,255]
# permute(1, 2, 0) 说明输入的tensor是 [C, H, W]
ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
im = Image.fromarray(ndarr)
im.save(fp, format=format)
```
所以

```python
import torchvision
import torch
from skimage import io
import numpy as np

image_path = r'D:\git\learn_python\images\27e1735ce8892eeafa202335c4388c0703a8c7f7511c00b88dbdc10566970104.png'
# [H, W, C]
image = io.imread(image_path)
# [0, 1]
image = image / 255
# [C, H, W]
image = image.transpose(2, 0, 1)

# gt_img 需要是 tensor 格式
gt_img = torch.from_numpy(image)
torchvision.utils.save_image(gt_img, '1.png')
```

```python