- [1. 总结](#1-总结)
- [2. 当函数用](#2-当函数用)
- [3. compose](#3-compose)
- [4. 例子](#4-例子)


---

## 1. 总结
```python
import torchvision.transforms as transforms

transforms.Resize((224, 224))

transforms.RandomCrop(32, padding=4),

transforms.RandomHorizontalFlip(),

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
transforms.ToTensor()

# 转化为图像, 可以被 plt.imshow() 显示图像
transforms.ToPILImage()

transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
```

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