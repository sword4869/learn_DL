- [1. install](#1-install)
- [2. Usage](#2-usage)
  - [2.1. 轮廓](#21-轮廓)
  - [2.2. 缩放](#22-缩放)

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
# 数组模式
h,w,c = image.shape
# ndarry
```

```python
# 需要是 uint8, np.uint16, np.float32
# 因为 np的float默认是np.float64，会报错
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
```

```python
# diff_gray 必须是灰度图
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