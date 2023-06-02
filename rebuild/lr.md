
```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# 调整 optimizer 的学习率 lr
# Decays the learning rate of each parameter group by gamma every epoch. 
# When last_epoch=-1, sets initial lr as lr.
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995) 

def train():
    for epoch in range(epoch_num):
        train_batch()

        # scheduler.step()通常用在epoch里面, 每调用step_size一次，对应的学习率就会按照策略调整一次。
        scheduler.step()

        val_batch()
```
## optimizer
```python
torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
torch.optim.AdamW(model.parameters(), lr=lr)
```
## lr_scheduler
```python
# 指数
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
# 线性步长下降
torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
```