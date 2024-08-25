[toc]

## 1. loss

- PSNR↑:
  - PSNR高于40dB说明图像质量几乎与原图一样好；
  - 在30-40dB之间通常表示图像质量的失真损失在可接受范围内；
  - 在20-30dB之间说明图像质量比较差；
  - PSNR低于20dB说明图像失真严重。
- SSIM↑：[0, 1]，越大代表图像越相似。如果两张图片完全一样时，SSIM值为1
- LPIPS↓

> 如何把 SSIM 用到loss里（如何把越大越好的指标用到越小越好的loss里）？

就是所谓的 D-SSIM = 1 - SSIM。SSIM的最大范围 - SSIM值。

```python
loss = (1.0 - ssim(image, gt_image))
```

## 2. overview

```python
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch
import numpy as np


def metrics3(imgs1, imgs2, channel_axis_imgs1, channel_axis_imgs2, device='cpu'):
    """Return psnr, ssim, lpips

    Args:
        imgs1 (NDarray or Tensor): single img or multiple imgs. 
            Because lpip does not process grayscale images, it does not consider grayscale images. Shape can be (N, 3, H, W)/(3, H, W), (N, H, W, 3)/(H,W,3). 
        imgs2 (NDarray or Tensor): 
        channel_axis_imgs1(int): 
        channel_axis_imgs2(int): 
        device (str, optional): For lpip model, it is about device you want to use rather than device the imgs locate. Defaults to 'cpu'.

    Return:
        (1, 1, 1) for single img or multiple imgs. 
    """
    def ret_np_torch(imgs, channel_axis):
        if type(imgs) == np.ndarray:
            imgs_np = np.array(imgs, dtype=np.float32)
        elif type(imgs) == torch.Tensor:
            imgs_np = np.array(imgs.detach().cpu().numpy(), dtype=np.float32)
        else:
            raise TypeError

        # single or multiple
        if len(imgs_np.shape) == 3:    
            imgs_np = imgs_np.transpose(channel_axis, (channel_axis + 1)%3, (channel_axis + 2)%3)
        else:
            imgs_np = imgs_np.transpose(0, channel_axis, channel_axis %3 + 1, channel_axis %3 + 2)

        imgs_tensor = torch.tensor(imgs_np, device=device, dtype=torch.float32)

        return imgs_np, imgs_tensor

    imgs1_np, imgs1_tensor = ret_np_torch(imgs1, channel_axis_imgs1)
    imgs2_np, imgs2_tensor = ret_np_torch(imgs2, channel_axis_imgs2)
    
    psnr = peak_signal_noise_ratio(imgs1_np, imgs2_np, data_range=1)

    # single or multiple
    if len(imgs1_np.shape) == 3:    
        ssim = structural_similarity(imgs1_np, imgs2_np, data_range=1, channel_axis=0)
    else:
        ssim = structural_similarity(imgs1_np, imgs2_np, data_range=1, channel_axis=1)

    torch_lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    lpips_metric = torch_lpips_model(imgs1_tensor, imgs2_tensor)

    return psnr, ssim, lpips_metric.detach().cpu().numpy().item()
```

## 3. psnr和ssim
> old name:
- `skimage.measure.compare_psnr` -> `skimage.metrics.peak_signal_noise_ratio`
- `skimage.measure.compare_ssim` -> `skimage.metrics.structural_similarity`

> data_range

[0, 255] 的 uint8, [0.0, 1.0] 的 float 的ndarry都行。

只要shape相同, 建议[H,W,3]。

### 3.1. data_range

The data range of the input image (distance between minimum and maximum possible values). By deault, 255 for uint8, 1 for float.

当是float时，最好还是都指定, `data_range=1`. 

- `dtype_range`

    ```python
    from skimage.util.dtype import dtype_range
    
    dtype_range[a.dtype.type]
    # np.uint8 (0, 255)
    # np.float32 (-1, 1)
    # np.float64 (-1, 1)
    ```
- psnr：是255和1
    ```python
    # psnr内部代码
    # 由于 true_min 的判断，所以没问题。255 for uint8, 1 for float
    if true_min >= 0:
        data_range = dmax
    else:
        data_range = dmax - dmin
    ```
- ssim：是255和2

    If `data_range` is not specified, the range is automatically guessed based on the image data type. However for floating-point image data, this estimate yields a result double the value of the desired range, as the `dtype_range` in `skimage.util.dtype.py` has defined intervals from `-1` to `+1`. 
    
    This yields an estimate of 2, instead of 1, which is most often required when working with image data (as negative light intentsities are nonsensical).

    ```python
    # ssim内部代码
    # 255 直接就是，没问题
    # 但是 np.float是-1，算下来就是 2，所以需要指定 data_range=1
    dmin, dmax = dtype_range[im1.dtype.type]
    data_range = dmax - dmin
    ```
### 3.2. psnr

可以看出，只要shape相同，[H,W,C],[C,H,W],[H*W,C]等等都行。
```python
from skimage.metrics import peak_signal_noise_ratio

# [0.0, 1.0] data_range=1., or [0, 255] data_range=255.
psnr = peak_signal_noise_ratio(img_ndarry, img2_ndarry, data_range=1)
```
上面的内部展开就是下面的psnr
```python
# [0.0, 1.0] data_range=1., or [0, 255] data_range=255.
def psnr(p0, p1, data_range=255.):
    return 10 * np.log10( data_range**2 / np.mean((1.*p0 - 1.*p1)**2) )
```
写成 lambda  的形式
```python
# [0.0, 1.0]
img2mse_torch = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr_torch = lambda x : 10. * torch.log10(1.0 / x)
# mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


# [0.0, 1.0]
img2mse_np = lambda x, y : np.mean((x - y) ** 2)
mse2psnr_np = lambda x : 10. * np.log10(1.0 / x)
# mse2psnr = lambda x : -10. * np.log(x) / np.log(np.array([10.]))


mse = img2mse(p0, p1)
mse2psnr(mse)
```
![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062018361.png)  

### 3.3. ssim

- `data_range` unit8还是float
    ```python
    # imageA, imageB 是 float
    ssim = structural_similarity(imageA, imageB, data_range=1)
    ```
    ```python
    # imageA, imageB 是 unit8
    ssim = structural_similarity(imageA, imageB, data_range=255)
    ```

- `channel_axis` 灰度图还是彩色图
    ```python
    # imageA, imageB 是 灰度图
    ssim = structural_similarity(imageA, imageB, data_range=1)
    ```
    CHW也行，改channel_axis就行。
    ```python
    # imageA, imageB 是 彩色图
    ssim = structural_similarity(imageA, imageB, data_range=1, channel_axis=2)
    ```
- `full` 差别图：
    ```python
    # full=True
    # 无论imageA, imageB是uint8还是float，diff返回的都是 [0,1.0]的float类型
    score, diff = structural_similarity(imageA, imageB, data_range=1, channel_axis=2, full=True)
    ```

```python
'''
compute the Structural Similarity Index (SSIM) between the two images
'''
import cv2
import numpy as np
from skimage.metrics import structural_similarity

# load the two input images
imageA = cv2.imread(r'D:\git\NeuLF\result\Exp_t_fern\train\epoch-870\gt.png')
imageB = cv2.imread(r'D:\git\NeuLF\result\Exp_t_fern\train\epoch-870\recon.png')

imageA = imageA / 255
imageB = imageB / 255

# full=True：添加一个返回对象，图片的差别
score, diff = structural_similarity(imageA, imageB, data_range=1, channel_axis=2, full=True)
print("SSIM: {}".format(score))

diff = (diff * 255).astype(np.uint8)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# 下面框选轮廓只是在diff的灰度图上进行，因而不是准的
# threshold the difference image, followed by finding contours to obtain the regions of the two input images that differ
thresh = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2  else cnts[1]

mask = np.zeros(imageA.shape, dtype='uint8') 
filled_mask = imageA.copy()

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.drawContours(mask, [c], 0, (0,255,0), -1)
    cv2.drawContours(filled_mask, [c], 0, (0,255,0), -1)

# show the output images
cv2.imshow("imageA", imageA)
cv2.imshow("imageB", imageB)
cv2.imshow("diff", diff)
cv2.imshow("diff_gray", diff_gray)
cv2.imshow("thresh", thresh)
cv2.imshow("mask", mask)
cv2.imshow("filled_mask", filled_mask)
cv2.waitKey(0)
```
可以看出用灰度图框选不太准（毕竟三通道融合了）

## 4. lpips
两种方式
- `lpips.LPIPS`
- `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`

区别：
- `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`就是在调用`lpips.LPIPS`，所以`normalize=True`的特性是一样的。
- `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`是对一个batch（即N)，求得一个整体平均的lpips。
- `lpips.LPIPS`是对一个batch（即N)，求得N个每张图片的lpips。

要求：
- 需要输入的是两个`tensor`, 不能ndarry
- `dtype=torch.float32`。float32是因为从numpy直接转化过来是float64，会报错。
  
    **float32 要么在 np.astype() , 要么在 tensor.dtype 处**

- device一致。vgg模型默认是在cpu上，因而如果传入gpu会出现设备的不一致问题.
  
    所以要么默认模型，则imgs是cpu上；要么让模型放在和imgs的gpu上。
- 建议[0, 1.0]的图片, `normalize=True`


```python
import lpips
import torch
import numpy as np

# 实验复现
_ = torch.manual_seed(123)
np.random.seed(123)

# float32 要么在 np.astype() , 要么在 tensor.dtype 处
imgs1 = np.random.rand(10, 100, 100, 3).astype(np.float32)
imgs1 = torch.tensor(imgs1, device='cuda:0').permute(0, 3, 1, 2)
imgs2 = np.random.rand(10, 100, 100, 3)
imgs2 = torch.tensor(imgs2, dtype=torch.float32, device='cuda:0').permute(0, 3, 1, 2)


############# `lpips.LPIPS`
# pip install lpips
lpips_model = lpips.LPIPS(net='vgg').to('cuda:0')
# 默认normalize=False needs the images to be in the [-1, 1] range. normalize=True is in the [0, 1] range.
# 默认reduction = 'mean', 还可以'sum'
l2 = lpips_model(imgs1, imgs2, normalize=True) 
print(l2.shape)
# torch.Size([10, 1, 1, 1])
print(l2.squeeze())
# tensor([0.3591, 0.3495, 0.3468, 0.3650, 0.3479, 0.3384, 0.3375, 0.3342, 0.3376,
#         0.3680], device='cuda:0', grad_fn=<SqueezeBackward0>)
print(l2.mean())
# tensor(0.3484, device='cuda:0', grad_fn=<MeanBackward0>)


############# `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`
# pip install torchmetrics[image] 
# pip install lpips
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
torch_lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to('cuda:0')
l = torch_lpips_model(imgs1, imgs2)
print(l)
# tensor(0.3484, device='cuda:0', grad_fn=<SqueezeBackward0>)
```