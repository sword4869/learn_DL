- [1. torch.cuda.OutOfMemoryError: CUDA out of memory](#1-torchcudaoutofmemoryerror-cuda-out-of-memory)
- [2. cache技巧](#2-cache技巧)
- [3. gradient\_accumulation\_steps](#3-gradient_accumulation_steps)
  - [3.1. manually](#31-manually)
  - [3.2. accelerate](#32-accelerate)


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

![图 6](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062011617.png)  

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
# `del` 只能删除内存里的变量
# GPU上的只能通过 `torch.cuda.empty_cache()`

del pipeline
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


## 3. gradient_accumulation_steps

我们有时被迫在训练过程中使用更小的 batch_size，这可能会导致收敛速度减慢和精度降低————解法就是梯度累计。 


假设原来的 batch size=10, 数据总量为1000，那么一共需要 100 train steps，同时**一共进行100次梯度更新**。

若是显存不够，我们需要减小 batch size ，我们设置gradient_accumulation_steps=2，那么我们新的 batch size=10/2=5，我们需要运行两次，才能在内存中放入10条数据，**梯度更新的次数不变为100次**，那么我们的train steps=200

### 3.1. manually

```python
train_dataloader_10 = torch.utils.data.DataLoader(
    dataset, batch_size=10, shuffle=True
)

train_dataloader_5 = torch.utils.data.DataLoader(
    dataset, batch_size=5, shuffle=True
)

def train(data_loader=train_dataloader_10):
    for epoch in range(num_epoch):
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs) 
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def train_gradient_accumulation_steps(data_loader=train_dataloader_5):
    gradient_accumulation_steps = 2
    for epoch in range(num_epoch):
        for i, batch in enumerate(data_loader):     # <<< smaller batch_size dataloader
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs) 
            loss = loss_function(outputs, targets)
            loss /= gradient_accumulation_steps     # <<< normalize loss
            loss.backward()

            # Gradient accumulation:                # <<< gradient_accumulation_steps
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()
```

- normalize loss

    Divide the running loss by `gradient_accumulation_steps`. This normalizes the loss to reduce the contribution of each mini-batch we are actually processing. 
    
    Depending on the way you compute the loss, you might not need this step: if you average loss within each batch, the division is already correct and there is no need for extra normalization.

- Please also note that some network architectures have batch-specific operations. 

    For instance, **batch normalization** is performed on a batch level and therefore may yield slightly different results when using the same effective batch size with and without gradient accumulation. This means that you should not expect to see a 100% match between the results.

### 3.2. accelerate
Accelerator is able to keep track of the batch number you are on and it will automatically know whether to step through the prepared optimizer and how to adjust the loss.

```python
from accelerate import Accelerator
accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader_10
)

for batch in train_dataloader:
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```
```python
from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=2)    # <<< gradient_accumulation_steps

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader_5
)                                           # <<< smaller batch_size dataloader
for batch in train_dataloader:            
    with accelerator.accumulate(model):     # <<< context manager warp model
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```