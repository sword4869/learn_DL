- [1. 总结](#1-总结)
- [2. 当函数用](#2-当函数用)
- [3. compose](#3-compose)
- [4. 例子](#4-例子)


---

## 1. 总结
```python
import torchvision.transforms as transforms

# 将图片的短边缩放成size的比例，然后长边也跟着缩放，使得缩放后的图片相对于原图的长宽比不变。
transforms.Resize(224),
# resize成自己想要的图片大小，可以直接使用transforms.Resize((H,W))
transforms.Resize((224, 224)),
transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
# 将原图片裁剪出一块随机位置和随机尺寸的图片,再缩放成相应 (size*size) 的比例
transforms.RandomResizedCrop(224),

#---------------------- (data augmentation)
# 从图片中心开始沿两边裁剪，裁剪后的图片大小为（size*size）
# 允许超过原尺寸，自动补充黑边
transforms.CenterCrop(32),
# 不允许超过原始尺寸+2倍的padding，补充黑边
transforms.RandomCrop(32, padding=4),

transforms.RandomHorizontalFlip(),
# 这里的效果是旋转5度
transforms.RandomAffine(5),
#----------------------

# H,W,C -> C,H,W
# uint8 -> tensor.float32 (0, 1)
transforms.ToTensor(),

# 转化为图像, 可以被 plt.imshow() 显示图像
transforms.ToPILImage(),

#----------------------


# 首先必须是 Tensor Image: 经过transforms.ToTensor(),
transforms.Normalize([0.5], [0.5]),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
),
```

PS: `transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),`

## 2. 当函数用

```python
X, y = mnist_train_totensor[0]
# 转化为图像, 可以被 plt.imshow() 显示图像
trans = transforms.ToPILImage()
X = trans(X)
print(type(X))
# PIL.Image.Image
```
## 3. compose
```python
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```
    Compose(
        Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=None)
        ToTensor()
    )

## 4. 例子

一般训练集用下data augmentation，动作多点，测试集自然不用那么多。

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
```