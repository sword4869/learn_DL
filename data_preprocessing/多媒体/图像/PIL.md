- [1. open,save,show](#1-opensaveshow)
- [2. convert](#2-convert)
- [3. 通道分离合并](#3-通道分离合并)
- [4. crop/paste](#4-croppaste)
- [5. rotate](#5-rotate)
- [6. resize](#6-resize)
- [7. 像素操作](#7-像素操作)
- [8. 和 Numpy 数组之间的转化](#8-和-numpy-数组之间的转化)

---

```bash
pip install pillow
```

## 1. open,save,show
```python
from PIL import Image

##################################### 打开
img = Image.open('../image/20220923201825.jpg')


##################################### 保存
img.save('1.jpg')

##################################### 图像显示

# 图片编辑器显示
img.show()

# matplotlib
import matplotlib.pyplot as plt
plt.imshow(img)

# 也可以直接在jupyter notebook 显示
img
```

## 2. convert
```python
'''
没有shape属性
- mode 属性为色彩模式，RGB 代表彩色图像，L 代表光照图像也即灰度图像等
- format 属性为图片的格式，如常见的 PNG、JPEG 等
'''

print(img.mode)     # RGB
print(img.format)   # JPEG


##################################### 色彩模式之转化灰度图
im = img.convert('L')       # 灰度图
# im = img.convert('RGB')   # RGB
# 如果是RGB的png，那么Image.open打开就是RGB。这时候想要RGBA，就convert强行加一个都是1的alpha
# im = img.convert('RGBA')   # RGBA
```
## 3. 通道分离合并
```python
##################################### 通道分离合并

# 分离
r, g, b = img.split()   # 3个灰度图
print(r.mode)    # L

fig, axs = plt.subplots(1, 3)
axs[0].imshow(r, cmap='gray')
axs[1].imshow(g, cmap='gray')
axs[2].imshow(b, cmap='gray')

# 合并
im = Image.merge('RGB', (r, g, b))
```

## 4. crop/paste

```python
##################################### 获取局部区域（裁剪）
# 左上角 X 坐标、Y 坐标，右下角 X 坐标、Y 坐标
box = (0, 0, 30, 30)
im = img.crop(box)


##################################### 粘贴图片

img.paste(img.crop(box), (10, 10))  # 左上角坐标
```
## 5. rotate

```python
##################################### 旋转

# 平面翻转
im1 = img.transpose(Image.FLIP_LEFT_RIGHT) # 左右翻转
im2 = img.transpose(Image.FLIP_TOP_BOTTOM) # 上下翻转
im3 = img.transpose(Image.ROTATE_180)      # 180, 
im4 = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM) # 左右+上下翻转=180

fig, axs = plt.subplots(1, 4)
axs[0].imshow(im1)
axs[1].imshow(im2)
axs[2].imshow(im3)
axs[3].imshow(im4)

# 角度旋转
im = img.rotate(45)
```
- 旋转：

  ![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062021749.png)  
  ![图 2](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062021750.png)  

## 6. resize
```python
>>> image.size
(1334, 2048)
>>> image.width
1334
>>> image.height
2048
```

```python
# (w, h)
im = img.resize((300, 600))
im = img.resize((300, 600), resample=Image.NEAREST)
```

## 7. 像素操作
```python
img = Image.open('../image/20220923201825.jpg')

##################################### 读取像素：方式1
# 从下标0开始
rgb_tuple1 = img.getpixel((0, 0))   # (245, 249, 232)

# 图像是58*28, (x,y)坐标系, x对应宽度，y对应高度
rgb_tuple2 = img.getpixel((57, 27))

##################################### 读取像素：方式2

pixels = img.load()
rgb_tuple3 = pixels[1,2]

##################################### 修改像素（涂色）
# 涂一格
for i in range(10):
    for j in range(10):
        img.putpixel((i, j), (255, 255, 0))

# 整体增强
im = img.point(lambda i: i * 1.5) # 对每个像素值乘以 1.5

# 对某个通道二值化
im = img.split()
im = im[2].point(lambda i: i > 128 and 255) # 对 B 通道进行二值化
```

## 8. 和 Numpy 数组之间的转化
```python
import numpy as np
from PIL import Image

# Image -> nparry：[0, 255], uint8, [H, W, C]
img = Image.open('../image/20220923201825.jpg')
array = np.array(img)
print(array.shape)  # (28, 58, 3)

# nparry -> Image
# Image.fromarray 需要 ndarry 数组的格式 是[0, 255], uint8, [H, W, C]
# image = (image * 255).astype(np.uint8)
out = Image.fromarray(image)
```

单通道，不能有通道

```python
import numpy as np
from PIL import Image

# image = np.random.rand(100, 100, 1) 不行
image = np.random.rand(100, 100)
image = (image * 255).astype(np.uint8)

out = Image.fromarray(image)
out.save('test.png')
```