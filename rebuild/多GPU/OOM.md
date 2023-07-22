- [1. torch.cuda.OutOfMemoryError: CUDA out of memory](#1-torchcudaoutofmemoryerror-cuda-out-of-memory)
- [2. cache技巧](#2-cache技巧)


---
## 1. torch.cuda.OutOfMemoryError: CUDA out of memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 14.14 GiB 
(GPU 0; 11.00 GiB total capacity; 
71.43 MiB already allocated; 
9.17 GiB free; 
80.00 MiB reserved in total by PyTorch)
```

![图 6](../images/25e84d781bb54fea8f7d152c8ce7ad9b479d0d3030ccb6f1b046305015b25f20.png)  

- Total Capacity: 11GB
  - reserved by pytorch: 80MB
    - already allocated: 71.43MB
    - cached(临时变量之类的): 80MB-71.43MB=0.86MB
  - free space: 9.17GB
  - other apps space(显示器，游戏之类的): 11GB-80MB-9.17GB=1.02G 

②就是`Tried to allocate 14.14 GiB `，会从free space 申请到already allocated。如果空间够，则申请成功；不够就OOM。
①强制清空pytorch cache，扩展free space。对应代码就是
```python
# 放在申请操作处前
torch.cuda.empty_cache()
```

## 2. cache技巧

1. 如果中间变量只是为了传递一次，那么就不要无用的中间变量。
```python
# def forward(self, x):
#   out_1 = self.conv_1(x)
#   out_2 = self.conv_2(out_1)
#   out_3 = self.conv_3(out_2)
#   return out_3

def forward(self, x):
  x = self.conv_1(x)
  x = self.conv_2(x)
  x = self.conv_3(x)
  return x
```
2. Torch变量原地操作

```python
X = torch.randn(3,4)
Y = torch.randn(3,4)

# 不要这样
# X = X + Y

# 方法1： +=, -+, %=, *=, /=
X += Y

# 方法2：[:]
X[:] = X + Y
```

3. 每个批次用到变量时，再移动到GPU上，而不是直接全移过去
```python
# x,y = AllData
# x, y = x.cuda(), y.cuda()

for x,y in DataLoder:
    x, y = x.cuda(), y.cuda()
```


4. 激活函数，`nn.ReLU(inplace=True)`


PS：`del` 只能删除内存里的变量, GPU上的只能通过 `torch.cuda.empty_cache()`
```python
import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()
```