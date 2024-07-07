tensor

会裁剪上限

```python
# [C, H, W]
torchvision.utils.save_image(gt_img, '1.png')
```

```python
# (1) [1, C, H, W]
# [-1.0, 1.8] 👉 [0.0, 255]

import matplotlib.pyplot as plt
# 先检查，是否在[0,255]内
self.zbuffer.squeeze(0).permute(1, 2, 0).add(1).mul(100).min()
self.zbuffer.squeeze(0).permute(1, 2, 0).add(1).mul(100).max()

plt.imshow(self.zbuffer.squeeze(0).permute(1, 2, 0).add(1).mul(80).to("cpu", torch.uint8).numpy())
```

 [type转化爆炸.md](type转化爆炸.md) 

```python
# True, False, [1,H,W,1]
plt.imshow(mask.squeeze(0).permute(1, 2, 0).cpu().numpy())
```

```python
# [-1.0, 1.0] 👉 [0.0, 1.0]
plt.imshow(normal_images.squeeze(0).add(1).mul(0.5).permute(1, 2, 0).cpu().numpy())
```

