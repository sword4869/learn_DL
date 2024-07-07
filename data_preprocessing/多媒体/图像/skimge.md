- [1. install](#1-install)
- [2. read, show, save](#2-read-show-save)
- [3. 灰度图](#3-灰度图)
- [4. 缩放/上下采样](#4-缩放上下采样)


---
## 1. install
```bash
pip install scikit-image
```


## 2. read, show, save
```python
from skimage import io

###### 读入图片
# as_gray=True，变为灰度图
# 读取进来就是 unit8
image = io.imread(image_path)

###### 显示图片

# [1] plt
import matplotlib.pyplot as plt
plt.imshow(image)

# [2] show when non-interacitvate mode
io.imshow(image)
io.show()

###### 保存图片
io.imsave(filename, image)
```

```python
# a gray-image is [H,W], an RGB-image [H,W,3] and an RGBA-image [H,W,4].
img.shape
```

## 3. 灰度图
```python
from skimage import color
image_gray = color.rgb2gray(image)
```
## 4. 缩放/上下采样

- `resize()`好，灰度图和彩色图的参数不动, `rescale()`还得加`channel_axis`
- 输入的`image`均可uint8或者float。
- 返回的图片都是 float64.

单张灰度图片
```python
from skimage.transform import rescale, resize, downscale_local_mean

## 都是(H, W)

# 缩放系数
image_rescaled = rescale(image, 1.0/4, anti_aliasing=True)
image_rescaled = rescale(image, (1.0/4, 1.0/4), anti_aliasing=True)

# 缩放后的尺寸
image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4),
                       anti_aliasing=True)

# 没有 anti_aliasing
image_downscaled = downscale_local_mean(image, (4, 4))
```
单张RGB图片
```python
from skimage.transform import rescale, resize, downscale_local_mean

## 都是(H, W, C)

# 缩放系数
image_rescaled = rescale(image, 1.0/4, anti_aliasing=True, channel_axis=2)
image_rescaled = rescale(image, (1.0/4, 1.0/4), anti_aliasing=True, channel_axis=2)

# 缩放后的尺寸
image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4),
                       anti_aliasing=True)
# 没有 anti_aliasing
image_downscaled = downscale_local_mean(image, (4, 4))
```
多图：风险就是图片过多，image_rescaled_failure变量的内存炸了，还是单张吧
```python
# 把图像数量也除以系数了，效果就是一张图片是四张图片重影
# imgs = (27, 3840, 2160, 3)
# image_rescaled_failure = (7, 960, 540, 3)
image_rescaled_failure = rescale(imgs, 1.0/4, anti_aliasing=True, channel_axis=3)

# 所以要加上数量通道
image_rescaled = rescale(imgs, (1, 1.0/4, 1.0/4), anti_aliasing=True, channel_axis=3)



# (N, H, W)
image_resized = resize(image, (image.shape[0], image.shape[1] // 4, image.shape[2] // 4), 
                        anti_aliasing=True)
```

