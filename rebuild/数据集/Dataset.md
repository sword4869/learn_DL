- [1. 自定义数据集1](#1-自定义数据集1)
- [2. 自定义数据集2](#2-自定义数据集2)

---
## 1. 自定义数据集1
```python
# __getitem__和__len__是子类必须继承的。
class NumberDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        samples = list(range(1, 1501))
        # 数据集划分训练和测试
        if train:
            self.samples = samples[:len(samples)*0.6]
        else:
            self.samples = samples[len(samples)*0.6:]

    #  __len__: 实现len(dataset)返回整个数据集的大小。
    def __len__(self):
        return len(self.samples)

    # __getitem__: 用来获取一些索引的数据，使dataset[item]返回数据集中第item个样本
    def __getitem__(self, item):
        return self.samples[item]
```


## 2. 自定义数据集2

```
.
├── bulbasaur
├── charmander
├── data.csv
├── mewtwo
├── pikachu
└── squirtle
````
每个文件夹即是类， 每个类里有属于这个类的图片。


```python
import os
import glob
import csv
from PIL import Image


# __getitem__和__len__是子类必须继承的。
class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        # labelString: labelIndex
        self.labelPair = {}
        # 排序保证每次打开都是一样的顺序
        ordedDirs = sorted(os.listdir(self.root))
        for dirName in ordedDirs:
            # 跳过非文件夹, data.csv
            if not os.path.isdir(os.path.join(self.root, dirName)):
                continue
            # 用录入字典的大小作为增长的下标
            self.labelPair[dirName] = len(self.labelPair.keys())
        print(self.labelPair)

        imagePaths, labelIndexs = self.load_csv('data.csv')

        # 数据集划分训练和测试
        coefficient = 0.6
        separation = int(len(imagePaths)*coefficient)
        if train:
            self.imagePaths = imagePaths[:separation]
            self.labelIndexs = labelIndexs[:separation]
        else:
            self.imagePaths = imagePaths[separation:]
            self.labelIndexs = labelIndexs[separation:]

    """
    如果有已经存在的csv文件, 那么直接去读；否则, 先创建再读
    fileName: csv文件名, 在root下面
    """

    def load_csv(self, fileName):
        fileName = os.path.join(self.root, fileName)
        # 没有时创建csv文件
        if not os.path.exists(fileName):
            # 图片路径列表
            imagePaths = []
            # 文件夹, 即类名
            for dirName in self.labelPair.keys():
                # 此类下的不同格式的图片, png, jpg, jpeg
                imagePaths += glob.glob(os.path.join(self.root,
                                        dirName, '*.png'))
                imagePaths += glob.glob(os.path.join(self.root,
                                        dirName, '*.jpg'))
                imagePaths += glob.glob(os.path.join(self.root,
                                        dirName, '*.jpeg'))
            # 写入csv
            with open(fileName, 'w', newline='') as fp:
                writer = csv.writer(fp)
                for imagePath in imagePaths:
                    labelString = imagePath.split(os.sep)[-2]
                    labelIndex = self.labelPair[labelString]
                    writer.writerow([imagePath, labelIndex])

        imagePaths, labelIndexs = [], []
        with open(fileName, 'r') as fp:
            reader = csv.reader(fp)

            for (imagePath, labelIndex) in reader:
                labelIndex = int(labelIndex)
                imagePaths.append(imagePath)
                labelIndexs.append(labelIndex)

        assert len(imagePaths) == len(labelIndexs)
        return imagePaths, labelIndexs

    #  __len__:实现len(dataset)返回整个数据集的大小。

    def __len__(self):
        return len(self.imagePaths)

    # __getitem__用来获取一些索引的数据，使dataset[item]返回数据集中第item个样本
    def __getitem__(self, item):
        imagePath, labelIndex = self.imagePaths[item], self.labelIndexs[item]
        x = Image.open(imagePath).convert('RGB')
        if self.transform:
            transforms(x)

        y = torch.tensor(labelIndex)
        return x, y
```


```python
myDataset = MyDataset('/home/lab/Downloads/pokeman', train=True)
print(len(myDataset))
x, y = myDataset[0]
print(type(x))
print(y)
'''
699
<class 'PIL.Image.Image'>
tensor(0)
'''
```
