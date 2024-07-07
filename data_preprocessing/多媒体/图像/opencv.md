- [install](#install)
- [Usage](#usage)
  - [imread](#imread)
  - [窗口大小](#窗口大小)
  - [按键](#按键)
  - [不断更新图片](#不断更新图片)
  - [获取某点像素值](#获取某点像素值)
  - [imwrite](#imwrite)
  - [缩放](#缩放)
  - [拼接图片](#拼接图片)
  - [画图](#画图)
- [other](#other)
- [前景背景](#前景背景)
- [轮廓](#轮廓)
- [回调函数](#回调函数)

---
## install

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


## Usage

### imread

路径没有英文就直接读。
```python
import cv2

image_path = r'C:\Users\lab\git\learn_python\images\8a85432a391c379d6006ddf59e77df7fd138f2a80e115058abbd06431d727d30.png'
image = cv2.imread(image_path)
cv2.imshow("imageA", image)
cv2.waitKey(0)
```
`imread` 不支持中文（读中文路径的图片，返回 None），那只能用PIL库先读，再转给cv。

```python
from PIL import Image
import numpy as np
import cv2

def load_img(img_path: str):
    '''
    cv2.imread()的替代函数，支持中文路径
    '''
    img = Image.open(img_path)
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
```
### 窗口大小

```python
# 鼠标能调 和 resizeWindow 起作用是同步的，要么都能，要么不能调。
# windows实际效果就两种：都有状态栏（关闭、全屏、最小），只分能调和不能调。
# (1) WINDOW_AUTOSIZE(默认值), WINDOW_FULLSCREEN: 图像原大小
# (2) WINDOW_NORMAL, WINDOW_GUI_EXPANDED, WINDOW_FREERATIO, WINDOW_KEEPRATIO: 能自己调
cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow("result", 400, 300) # 设置窗口大小
cv2.moveWindow("winname",x,y)       # 设置窗口的位置，窗口左上角
```
关闭窗口
```python
cv2.destroyWindow(winname)

cv2.destroyAllWindows()
```
### 按键

```python
# waitKey单位ms，默认0表示无限等待，默认退出是esc
import cv2

img = cv2.imread('1.png')
cv2.imshow('image', img)
cv2.waitKey()
```

```python
if cv2.waitKey() == ord('q'):
    exit()
```

### 不断更新图片

只要同一个窗口就行，无须其他操作

```python
import cv2

img = cv2.imread('1.png')
for i in range(0, img.shape[1],10):
    cv2.line(img, (0,0),(i,i),(255,0,0),5)
    cv2.imshow('image', img)
    cv2.waitKey(500)
```

### 获取某点像素值

```python
pos = (833, 833)    # x, y
color = img[pos[1], pos[0]]    # h, w
print(color)        # [B G R]，而且是ndarry
# [224 224 224]         
```

### imwrite

img的数据类型是array（即数组类型），这里一般情况下要填入的是8位的单通道或3通道（带有BGR通道顺序）

imwrite函数是基于文件扩展名选择图像的格式：

▶对于PNG，JPEG2000和TIFF格式，可以保存16位无符号（CV_16U）图像。

▶32位浮点（CV_32F）图像可以保存为PFM，TIFF，OpenEXR和Radiance HDR格式; 使用LogLuv高动态范围编码（每像素4个字节）保存3通道（CV_32FC3）TIFF图像。

▶可以使用此功能保存带有Alpha通道的PNG图像。为此，创建8位（或16位）4通道图像BGRA，其中alpha通道最后。完全透明的像素应该将alpha设置为0，完全不透明的像素应该将alpha设置为255/65535。

```python
depth_min = depth.min()
depth_max = depth.max()

max_val = (2 ** (8 * bits)) - 1
if depth_max - depth_min > np.finfo("float").eps:
    out = (depth - depth_min) / (depth_max - depth_min) * max_val
else:
    out = np.zeros(depth.shape, dtype=depth.dtype)

if bits == 1:
    cv2.imwrite(path, out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
elif bits == 2:
    cv2.imwrite(path, out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
```

- `cv2.CV_IMWRITE_JPEG_QUALITY`：设置 `.jpeg/.jpg` 格式的图片质量，取值为 0-100（默认值 95），数值越大则图片质量越高；
- `cv2.CV_IMWRITE_WEBP_QUALITY`：设置 `.webp` 格式的图片质量，取值为 0-100；
- `cv2.CV_IMWRITE_PNG_COMPRESSION`：设置 `.png` 格式图片的压缩比，取值为 0-9（默认值 3），数值越大则质量越低。

```python
# 默认值为95%质量
>>> cv2.imwrite("F://1.jpeg",img)
# 无损版
>>> cv2.imwrite("F://2.jpeg",img, [cv2.IMWRITE_JPEG_QUALITY, 100])
# 战损版
>>> cv2.imwrite("F://3.jpeg",img, [cv2.IMWRITE_JPEG_QUALITY, 2])
# 多参数演示
>>> cv2.imwrite("F://5.jpeg",img, [cv2.IMWRITE_JPEG_LUMA_QUALITY, 10, cv2.IMWRITE_JPEG_QUALITY, 100])
``` 

### 缩放

```python
# 数组模式 ndarry
h,w,c = image.shape
```

`cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])`

```python
# 方法1：dsize
# 函数传入是特殊的 (W,H), 但是输出的图片还是 (H,W,C)
img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

# 方法2：fx, fy
# dsize需要设置None，一起用就会只用dsize起作用
img = cv2.resize(img, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
```

e.g.
```python
# 调整 pasteq 的大小
paste = cv2.resize(paste, (paste.shape[1]//4, paste.shape[0]//4), interpolation=cv2.INTER_AREA)
cv2.imshow('paste', paste)
cv2.waitKey()
```

### 拼接图片
```python
# 拼接图片
paste = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype=np.float32)
paste[:, :img.shape[1], :] = img
template_img = cv2.imread(os.path.join(output_image_dir, 'template.png'))
paste[:, img.shape[1]:, :] = template_img
```


### 画图

```python
# 线
cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
cv2.line(img, (0,0),(511,511),(255,0,0),5)

# 矩形
cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

# 圆形
cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
cv2.circle(img,(447,63), 63, (0,0,255), 3)

# 椭圆
# center：中心位置
# axes：轴长度（长轴长度，短轴长度）
# angle：椭圆在逆时针方向上的旋转角度
# startAngle：主轴顺时针方向测量的椭圆弧的起点
# endAngle：主轴顺时针方向测量的椭圆弧的终点
cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])
cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

# 多边形
cv2.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]])
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))     # 顶点数x1x2形状
cv2.polylines(img,[pts],True,(0,255,255))
```
thickness取-1是实心，否则空心。默认厚度 = 1.

lineType默认是`cv2.LINE_AA`抗锯齿线条，不用管。

```python
# 文字
# org：您想要放置它的位置坐标（即数据开始的左下角）
cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
cv2.putText(img,'OpenCV',(10,500), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2)
```

## other

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


## 前景背景

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

## 轮廓
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

## 回调函数

```python
# 固定格式: (event, x, y, flags, param)
def callback_click(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        print(f'click {x} {y}')
        cmd = f'adb shell input tap {x} {y}'
        subprocess.check_output(cmd, shell=True)

cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('image', callback_click)

while True:
    # 缩小显示
    cv_img = cv2.resize(cv_img, (cv_img.shape[1]// RESIZE, cv_img.shape[0]// RESIZE), interpolation=cv2.INTER_AREA)
    cv2.imshow('image', cv_img)
```