```python
# Decays the learning rate of each parameter group by gamma every epoch. 
# When last_epoch=-1, sets initial lr as lr.
self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995) 


def train(self):
    for epoch in range(epoch_num):
        train_batch()

        # scheduler.step()通常用在epoch里面
        self.scheduler.step()

        val_batch()
```

```python
# scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次。
torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
```