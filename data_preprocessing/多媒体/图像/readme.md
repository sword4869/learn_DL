- [1. 读取、保存、显示](#1-读取保存显示)
- [2. 多图](#2-多图)

---

## 1. 读取、保存、显示

- `[0, 255]`, `np.uint8`
- `[0.0, 1.0]`, `np.float64`

1. 读入的格式除了PIL，都是 NumPy arrays `<class 'numpy.ndarray'>`的`uint8`类型， shape 都是`(H,W,C)`. 

2. shape

    就PIL特殊，`img.size`是`(W,H)`

    但PIL有函数可以和ndarry转化，转化为ndarry的后就正常是`(H,W,C)`.

3. 颜色通道
    
    就`opencv`是BGR，其他都是RGB。

    RGB and BGR use the same color space, except the order of colors is reversed. 
    
    - 不论 float or uint8
  
        `image = image[:, :, ::-1]` 

    - 需要是uint8格式
        
        ```python
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ```

|库|读入|保存|显示|
|:-:|:-:|:-:|:-:|
|skimage|uint8|uint8|float,uint8都行|
|imageio|uint8|uint8|float,uint8都行|
|PIL|PIL转为ndarry后是uint8， <br> ndarry转为PIL需uint8|PIL格式|PIL格式|
|opencv|uint8, 但BGR|float或者uint8的range`[0,255]`都行，但BGR||
|torchvision||内部自动转化为PIL。但是通道有变, `[C, H, W]`||


```python
from skimage import io
import imageio.v3 as iio
from PIL import Image
import cv2
import numpy as np

image_path = '../images/27e1735ce8892eeafa202335c4388c0703a8c7f7511c00b88dbdc10566970104.png'
# 保存图片格式选择`png`。`jpg`会压缩，图片质量下降，这比较指标的时候就麻瓜了。
filename = './1.png'

# 读入
image1 = io.imread(image_path)     # uint8
image2 = iio.imread(image_path)    # uint8
image3 = Image.open(image_path)    # PIL
image4 = cv2.imread(image_path)    # uint8

# 保存
io.imsave(filename, image1)
iio.imwrite(filename, image2)
image3.save(filename)
cv2.imwrite(filename, image4)
```

skimage, imageio, torch 都是调用 PIL的`save()`。

```python
# float[0., 1.] -> uint8
# 不能缩写除法 image /= 255
# UFuncTypeError: Cannot cast ufunc 'divide' output from dtype('float64') to dtype('uint8') with casting rule 'same_kind'
image = image / 255.
# python 除法默认是浮点数, 4/2=2.0, 所以结果都是 float64 
image = image / 255


# float[0., 1.] -> uint8
image = (image * 255).astype(np.uint8)

# float[-1., 1.] -> uint8
image = image * 0.5 + 0.5
image = (image * 255).astype(np.uint8)
```

## 2. 多图

```python
import torchvision
from PIL import Image

def make_Tensor2PIL_grid(x):
    """
    Given a batch of images x, make a grid and convert to PIL
    x: Tensor(B,C,H,W) or a list of Tensors(C,H,W)
    return: a PIL image
    """
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_PIL2PIL_grid(images, size=64):
    """
    Given a list of PIL images, stack them together into a line for easy viewing
    images: a list of PIL images
    return: a PIL image
    """
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im
```