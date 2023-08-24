- [1. optimizer](#1-optimizer)
- [2. lr\_scheduler](#2-lr_scheduler)
- [3. weight\_decay](#3-weight_decay)
- [4. 联合优化](#4-联合优化)


---

## 1. optimizer
```python
torch.optim.SGD(model.parameters(), lr=lr)

# Adam优化器的主要吸引力在于它对初始学习率不那么敏感。
torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
```

## 2. lr_scheduler

Decays the learning rate of each parameter group by gamma **every epoch**. 
```python
# 指数
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# 线性步长下降
# step_size指示的是 iteration 吗
torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
```

> example
根据预设值初始化 optimizer 的学习率 lr，再用`lr_scheduler`调整 optimizer 的学习率 lr
```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995) 

def train():
    for epoch in range(epoch_num):
        train_batch()

        # scheduler.step()通常用在epoch里面, 每调用step_size一次，对应的学习率就会按照策略调整一次。
        scheduler.step()

        val_batch()
```
## 3. weight_decay
默认情况下，PyTorch同时衰减权重和偏移。
```python
torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
```

直接通过weight_decay指定weight decay超参数。这里我们只为权重设置了weight_decay，所以偏置参数不会衰减。
```python
# 偏置参数没有衰减
optimizer = torch.optim.SGD(
    [
        {"params": model[0].weight, 'weight_decay': wd},
        {"params": model[0].bias}
    ],
    lr=lr
)
```

## 4. 联合优化

```python
# 联合优化，转为list来连接不同模型的参数
grad_vars = list(model.parameters())
grad_vars += list(model_fine.parameters())

optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
```