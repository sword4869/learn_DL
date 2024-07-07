- [1. Net](#1-net)
  - [1.1. model 加载梯度与否](#11-model-加载梯度与否)
    - [1.1.1. 只关乎BN和Dropout](#111-只关乎bn和dropout)
    - [1.1.2. torch.no\_grad()](#112-torchno_grad)
  - [1.2. loss 和 optimizer 的三者顺序](#12-loss-和-optimizer-的三者顺序)
  - [1.4. 训练、验证、测试](#14-训练验证测试)
- [2. train\_val](#2-train_val)

---
## 1. Net

### 1.1. model 加载梯度与否

![图 7](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062010404.png)  

#### 1.1.1. 只关乎BN和Dropout

`model.train()`的作用是启用 Batch Normalization 和 Dropout。在train模式，Dropout层会按照设定的参数p设置保留激活单元的概率，如keep_prob=0.8，Batch Normalization层会继续计算数据的mean和var并进行更新。

`model.eval()`的作用是不启用 Batch Normalization 和 Dropout。在eval模式下，Dropout层会让所有的激活单元都通过，而Batch Normalization层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。

`model.eval()`不会影响各层的梯度计算行为，即会和训练模式一样进行梯度计算和存储，只是不进行反向传播。

也就是说，没有用BN和Dropout的架构，就不用写这个。

#### 1.1.2. torch.no_grad()

只是防止梯度传递，没有梯度只节省一点点内存，OOM还是会发生。

PS: 写成函数的形式
```python
@torch.no_grad()
def eval(args):
    pass

eval(args)
```

### 1.2. loss 和 optimizer 的三者顺序

`loss.backward()`紧跟着就是`optimizer.step()`来完成梯度更新，所以只要`optimizer.zero_grad()`不写在二者中间就行。

也就是说，`optimizer.zero_grad()`可以写在for循环开始，还可以写在for循环中间`optimizer.step()`的前面，也可以写在for循环最后面，即`optimizer.step()`后面。

```python
for batch in train_loader:
    optimizer.zero_grad()

    outputs = model() 
    loss = loss()

    train_loss.backward()
    optimizer.step()
---
for batch in train_loader:
    outputs = model() 
    loss = loss()

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
---
for batch in train_loader:
    outputs = model() 
    loss = loss()

    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

!!! note: `optimizer.zero_grad()`放在最后不会对第一个batch造成影响！

    因为现在的pytorch，模型（新创立的、load pretrained）参数的**梯度**初始化就是`None`, `optimizer.zero_grad()`也是把模型参数的梯度置`None`。所以没有区别。
    
    PS：[detach能够阻断梯度回传.md](./detach能够阻断梯度回传.md)这篇文章也是如此，pytorch更新后，就没有梯度问题了。
    
    PS：
    <details>
    <summary> code </summary>
    
    ```python
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from accelerate.utils import set_seed
    
    set_seed(42)
    
    class RangeDataset(Dataset):
        def __init__(self, length):
            self.len = length
            self.data = torch.arange(0, length, dtype=torch.float32)
    
        def __getitem__(self, index):
            return self.data[index]
    
        def __len__(self):
            return self.len


    class Model(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, 1000),
                nn.Linear(1000, output_size)
            )
    
        def forward(self, input):
            output = self.fc(input)
            return output


    def train(rank):
        # create model and move it to GPU with id rank
        model = Model(5, 2).to(rank)
        # DataLoader
        rand_dataset = RangeDataset(length=100)
        rand_dataloader = DataLoader(
            rand_dataset, batch_size=5)      # <<<
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        for data in tqdm(rand_dataloader, disable=(rank!=0)):
            inputs = data.to(rank)
            outputs = model(inputs)
            labels = torch.randn(outputs.shape).to(rank)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(model.fc[0].weight)       # 看看输出，保证不是nan，训练炸了
        print(model.fc[0].weight.grad)  # 看看输出，保证不是nan，训练炸了
        torch.save(model.state_dict(), f'./model_{rank}.pth')


    def train2(rank):
        # create model and move it to GPU with id rank
        model = Model(5, 2).to(rank)
        ckpt = torch.load("model_0.pth", map_location="cpu")
        model.load_state_dict(ckpt)
    
        # DataLoader
        rand_dataset = RangeDataset(length=100)
        rand_dataloader = DataLoader(
            rand_dataset, batch_size=5)      # <<<
        loss_fn = nn.MSELoss()
        # 因为是同步的进度，所以只显示一个进程的进度条就行
        for data in tqdm(rand_dataloader, disable=(rank!=0)):
            inputs = data.to(rank)
            outputs = model(inputs)
            labels = torch.randn(outputs.shape).to(rank)
            loss = loss_fn(outputs, labels)
            print(model.fc[0].weight)
            print(model.fc[0].weight.grad)      # None
            return
    train(0)
    train2(0)
    ```
    
    </details>


### 1.4. 训练、验证、测试

![图 5](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062010405.png)  


![图 4](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062010406.png)  


![图 6](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062010407.png)  


## 2. train_val

```python
def train(train_loader, val_loader, model):
    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x) 
            batch_loss = loss(outputs, y)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

            train_acc += (train_pred.cpu() == y.cpu()).sum().item()
            train_loss += batch_loss.item()
            
            batch_loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 



        # validation
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, (x,y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                batch_loss = loss(outputs, y) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == y.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

        print(
            f'[{epoch + 1:03d}/{num_epoch:03d}]',
            f'Train Acc: {train_acc/len(train_loader.dataset):3.6f}',
            f'Loss: {train_loss/len(train_loader):3.6f}',
            f'| Val Acc: {val_acc/len(val_loader.dataset):3.6f}',
            f'loss: {val_loss/len(val_loader):3.6f}'
        )

        # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'saving model with acc {best_acc/len(val_loader.dataset):.3f}')

train(train_loader, val_loader, model, )
```

```python
def test(test_loader, model, device):
    predict = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.to(device)
            outputs = model(x)
            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                predict.append(y)

    return predict

# reload the best model
del model
model = Classifier().to(device)
ckpt = torch.load(model_path, map_location='cuda:0') 
model.load_state_dict(ckpt)
predict = test(test_loader, model, device)
```