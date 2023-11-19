- [1. install](#1-install)
- [2. Usage](#2-usage)
  - [2.1. 轮廓](#21-轮廓)
  - [2.2. 缩放](#22-缩放)
  - [2.3. 前景背景](#23-前景背景)

---
## 1. install

a. Packages for standard desktop environments (Windows, macOS, almost any GNU/Linux distribution)

- Option 1 - Main modules package: 
    `pip install opencv-python`
- Option 2 - Full package (contains both main modules and contrib/extra modulesm listing from OpenCV documentation): 
    `pip install opencv-contrib-python` 

- Option 3 - Headless main modules package: 
    `pip install opencv-python-headless`

- Option 4 - Headless full package (contains both main modules and contrib/extra modules): 
    `pip install opencv-contrib-python-headless`

Headless: Packages for server environments (such as Docker, cloud environments etc.), no GUI library dependencies

所以，一般选择 `pip install opencv-contrib-python` 

## 2. Usage

```python
import cv2

image_path = r'C:\Users\lab\git\learn_python\images\8a85432a391c379d6006ddf59e77df7fd138f2a80e115058abbd06431d727d30.png'
image = cv2.imread(image_path)
cv2.imshow("imageA", image)
cv2.waitKey(0)
```

```python
paste = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype=np.float32)
paste[:, :img.shape[1], :] = img
template_img = cv2.imread(os.path.join(output_image_dir, 'template.png'))
paste[:, img.shape[1]:, :] = template_img
# 调整 pasteq 的大小
paste = cv2.resize(paste, (img.shape[1]//4, img.shape[0]//4))
cv2.namedWindow('paste', 0)
cv2.imshow('paste', paste.astype(np.uint8))
if cv2.waitKey(0) == ord('q'):
    exit()
```

```python
# 数组模式 ndarry
h,w,c = image.shape
```

```python
# 需要是 uint8, np.uint16, np.float32
# 因为 np的float默认是np.float64，会报错
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
```

```python
# diff_gray 必须是灰度图
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
```
### 2.1. 轮廓
```python
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2  else cnts[1]
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.drawContours(mask, [c], 0, (0,255,0), -1)
```
### 2.2. 缩放
`cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])`

```python
# 方法1：dsize
# 函数传入是特殊的 (W,H), 但是输出的图片还是 (H,W,C)
img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

# 方法2：fx, fy
# dsize需要设置None，一起用就会只用dsize起作用
img = cv2.resize(img, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
```

### 2.3. 前景背景


- img: 输入图像，支持8位3通道
- mask: 掩码图像，可以设置为:
    GCD_BGD(=0），背景；
    GCD_FGD(=1），前景；
    GCD_PR_BGD(=2），可能的背景；
    GCD_PR_FGD(=3），可能的前景。
- rect: 前景的矩形，格式为(x,y,w,h），分别为左上角坐标和宽度，高度
- bdgModel, fgdModel: 算法内部是用的数组，只需要创建两个大小为(1,65）np.float64的数组。
- iterCount: 迭代次数
- mode:使用矩阵模式还是蒙板模式
    cv2.GC_INIT_WITH_RECT
    cv2.GC_INIT_WITH_MASK
```python
import numpy as np
import cv2
# 读取图像
img = cv2.imread(r'C:\Users\lab\Pictures\1111.png')

# ROI 
r = cv2.selectROI('input', img, False)  # 返回 (x_min, y_min, w, h)
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
# 手动
rect = (1, 1, img.shape[1], img.shape[0]) # (x_min, y_min, w, h)

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img_f = img * mask2[:, :, np.newaxis]
img_bg = img * (1 - mask2[:, :, np.newaxis])

# 显示图像
cv2.imshow("img", img)
cv2.imshow("img_f",img_f)
cv2.imshow("img_bg",img_bg)
cv2.waitKey()
cv2.destroyAllWindows()
```