- [1. 提取 alpha 通道](#1-提取-alpha-通道)
- [2. 混合前景背景](#2-混合前景背景)

---

## 1. 提取 alpha 通道

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = np.array(Image.open(r'D:\Downloads\noddles.png'))
# plt 和 opencv 对于 uint8 还是 float 都行
# img = np.array(img) / 255.0

plt.imshow(img[..., 3:4], cmap='gray')
####### plt则需要 Third dimension must be 3 or 4
plt.imsave('alpha.png', img[..., 3:4].repeat(3, axis=2))
# img = np.array(Image.open(r'alpha.png'))
# print(img.shape)  
# (4012, 4100, 4)


###### opecv 可以直接单通道保存, 保存结果就是1
import cv2
cv2.imwrite('alpha2.png', img[..., 3:4])     
# img = np.array(Image.open(r'alpha2.png'))
# print(img.shape)  
# (4012, 4100)
```

## 2. 混合前景背景
$\mathbf {c}_{\text {f}} * \mathbf {\alpha}_{\text {f}} + \mathbf {c}_{\text {b}} * \mathbf {\alpha}_{\text {b}}$

通常：$\mathbf {\alpha}_{\text {b}} = 1 - \mathbf {\alpha}_{\text {f}}$，背景的不透明度部分就是前景的透明部分。

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = np.array(Image.open(r'D:\Downloads\noddles.png'))
img = np.array(img) / 255.0

# img2: [h, w, 3]
img2 = img[..., :3] * img[..., 3:4] + bg_color * (1.0 - img[..., 3:4])

# 背景纯白
# (1.0 - img[..., -1:]) 其实是 (1.0 - img[..., -1:]) * 1, 1是nparray的广播操作就当于rgb255白色。
img2 = img[..., :3] * img[..., 3:4] + (1.0 - img[..., 3:4])

# 背景纯黑
img2 = img[..., :3] * img[..., 3:4]
```

`img = np.array(img) / 255.0` 必须，因为RGBA都是0-255的话，(0-255)x(0-255), 那么结果就超了255，而如果是 (0-1)x(0-1)，那么结果就没有问题。
