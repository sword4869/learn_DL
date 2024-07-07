
《2015. U-net: Convolutional networks for biomedical image segmentation》

原本是用于医学图像分割。后来被用于图像分隔。现在是diffusion。

## 架构

![168967377700130926146.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012060.png)

将 a contracting path (left side) and an expansive path (right side) 的四级的不同尺度的特征融合起来。

每个3x3的卷积都紧跟着ReLU，特征融合的左边都是ReLU后的结果。

左边：
- 2个3x3的卷积（**特征通道C的加倍**）：
    第一个卷积就完成特征通道的加倍，第二个卷积对特征通道就不变了
    原始图片特征通道数-> f -> fx2 -> fx4 -> fx8
- 卷积完后的特征将会传给右边。
- 2x2的最大池化（**图片尺寸HW的下采样**）

中间的bottle-neck：
- 2个3x3的卷积：fx8 -> fx16

右边：
- 2x2的 up-convolution / deconvolution（**图片尺寸HW的上采样，并且通道数C减半**）：
  并不是只想最大池化一样只修改尺寸。
  bottle-neck 的 fx16 -> fx8 ->fx4 -> fx2 -> f
- concatenate 灰色的操作:  
    在通道维度上拼接。
    左边卷积后的和右边up-convolution后的，通道数双倍了。fx16 -> fx8 ->fx4 -> fx2
- 2个3x3的卷积（**特征通道C的减半**）：fx8 ->fx4 -> fx2 -> f

最后：
- 1x1的卷积：输出图片的通道。f -> 输出图片特征通道数


Segmentation of a 512x512 image takes less than a second on a recent GPU.

## 具体实现

原本的unet是padding=0，从而图片尺寸越来越小，而且拼接的时候还得crop对齐；现在都是用half padding保持原图尺寸。

![168967377700130926822.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012061.png)



![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012062.png)  

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class Up_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.D1A = Double_conv2d_bn(1, 8)
        self.D2A = Double_conv2d_bn(8, 16)
        self.D3A = Double_conv2d_bn(16, 32)
        self.D4A = Double_conv2d_bn(32, 64)

        self.D5 = Double_conv2d_bn(64, 128)

        self.D4B = Double_conv2d_bn(128, 64)
        self.D3B = Double_conv2d_bn(64, 32)
        self.D2B = Double_conv2d_bn(32, 16)
        self.D1B = Double_conv2d_bn(16, 8)

        self.U4 = Up_conv2d_bn(128, 64)
        self.U3 = Up_conv2d_bn(64, 32)
        self.U2 = Up_conv2d_bn(32, 16)
        self.U1 = Up_conv2d_bn(16, 8)

        self.layer_out = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # left side
        d1a = self.D1A(x)
        pool1 = F.max_pool2d(d1a, 2)

        d2a = self.D2A(pool1)
        pool2 = F.max_pool2d(d2a, 2)

        d3a = self.D3A(pool2)
        pool3 = F.max_pool2d(d3a, 2)

        d4a = self.D4A(pool3)
        pool4 = F.max_pool2d(d4a, 2)

        # bottom
        d5 = self.D5(pool4)

        # right side
        u4 = self.U4(d5)
        concat4 = torch.cat([d4a, u4], dim=1)
        d4b = self.D4B(concat4)

        u3 = self.U3(d4b)
        concat3 = torch.cat([d3a, u3], dim=1)
        d3b = self.D3B(concat3)

        u2 = self.U2(d3b)
        concat2 = torch.cat([d2a, u2], dim=1)
        d2b = self.D2B(concat2)

        u1 = self.U1(d2b)
        concat1 = torch.cat([d1a, u1], dim=1)
        d1b = self.D1B(concat1)

        # out
        outp = self.layer_out(d1b)
        outp = self.sigmoid(outp)
        return outp


model = Unet()
X = torch.rand(10, 1, 256, 256)
Y = model(X)
print(Y.shape)
# torch.Size([10, 1, 256, 256])
```