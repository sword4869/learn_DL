- [1. gpu可用](#1-gpu可用)
- [2. torch.device](#2-torchdevice)
- [3. 默认GPU](#3-默认gpu)
- [4. 默认tensor类型](#4-默认tensor类型)
- [5. 迁移](#5-迁移)
  - [5.1. to() 和 cuda()](#51-to-和-cuda)
  - [5.3. 张量初始化设备](#53-张量初始化设备)

---
## 1. gpu可用
```python
import torch

print(torch.cuda.is_available())
# True
print(torch.cuda.device_count())
# 2

```
## 2. torch.device
```python
# 分配CPU
d1 = torch.device('cpu')

# 分配默认GPU，而不是表示第一块GPU
d2 = torch.device('cuda')

# 分配第二块GPU
d3 = torch.device('cuda:1')
d4 = torch.device('cuda', 1)

d5 = torch.device(0)      # torch.device('0')不行; 没有 torch.device()
d5 = torch.device(1)

# 默认GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 第二块GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```

## 3. 默认GPU

```python
#### 修改默认gpu
torch.cuda.set_device(1)

X = torch.ones(2, 3).cuda()
print(X.device)
# cuda:1

device = torch.device("cuda")
print(device)       # 这里虽然输出 cuda，但并不表示 cuda:0，其实是 cuda:1
# cuda
X = torch.ones(2, 3).to(device)
print(X.device)
# cuda:1
# 一个容易出错的细节：`device = torch.device("cuda")`换成`device = torch.device("cuda:0")`
#   就只能是cuda:0了


#### 还是可以指定
X = torch.ones(2, 3).cuda(0)
print(X.device)
# cuda:0
X = torch.ones(2, 3).to(0)
print(X.device)
# cuda:0
```

```python
with torch.cuda.device(1):
    X = torch.ones(2, 3).cuda()
    
    # 效果一样，修改默认gpu
    device = torch.device("cuda")
    X = torch.ones(2, 3).to(device)
```
## 4. 默认tensor类型

```python
import torch
from torch import nn

torch.set_default_tensor_type('torch.cuda.FloatTensor')     # torch.cuda.DoubleTensor 双精度

# 张量
X = torch.ones(2, 3)
print(X.device)
# cuda:1

net = nn.Sequential(nn.Linear(3, 1))
print(net[0].weight.data.device)
# cuda:1

print(net(X).device)
# cuda:1
```
## 5. 迁移

### 5.1. to() 和 cuda()
```python
torch.cuda.set_device(1)

Y = torch.rand(2, 3)                # cpu


# 无参 cpu
Y2 = Y.to()                         # cpu


Y3 = Y.to('cpu')                    # cpu
Y4 = Y.to(0)                        # cuda:0
Y5 = Y.to('cuda')                   # cuda:1
Y6 = Y.to('cuda:0')                 # cuda:0
Y7 = Y.to(torch.device('cuda:0'))   # torch.device 类型
```

```python
net = nn.Sequential(nn.Linear(3, 1))
# 没有 `net.device`, 只能从模型参数看出模型在哪里
print(net[0].weight.data.device)
# cpu
```

除了无参是默认GPU和没有`cuda('cpu')`，其他效果一样

```python
torch.cuda.set_device(1)

Y = torch.rand(2, 3)                # cpu


# 默认gpu
Y2 = Y.cuda()                         # cuda:1

# 指定gpu
Y3 = Y.cuda(0)                        # cuda:0
Y4 = Y.cuda('cuda')                   # cuda:1
Y5 = Y.cuda('cuda:0')                 # cuda:0
Y6 = Y.cuda(torch.device('cuda:0'))   # torch.device 类型
```
PS: 张量和模型还有个区别是返回值的写法
- 张量必须`a = a.to(0)`
- 模型可以直接 `model.to(0)` 和 `model = model.to(0)`
```python
>>> import torch
>>> a = torch.randn(2,2)
>>> a.cuda()                                      # 临时而已
tensor([[ 0.6088, -2.2590],
        [ 0.0765,  0.4449]], device='cuda:0')
>>> a
tensor([[ 0.6088, -2.2590],
        [ 0.0765,  0.4449]])
        
>>> model = torch.nn.Linear(2,2)
>>> model.cuda()                                  # 已经在GPU上
Linear(in_features=2, out_features=2, bias=True)
>>> model.weight.data.device
device(type='cuda', index=0)
```

### 5.3. 张量初始化设备

只有张量有这个属性，网络不能这样指定
```python
# 可以直接写，['cpu', 0, 'cuda', 'cuda:0', torch.device类型]
X = torch.ones(2, 3, device='cpu')
Y = torch.ones(2, 3, device=torch.device('cpu'))
```