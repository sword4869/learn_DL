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


```python
>>> optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
>>> optimizer.state_dict()
{
    'state': {}, 
    'param_groups': [
        {
            'lr': 0.001, 
            'momentum': 0.9, 
            'dampening': 0, 
            'weight_decay': 0, 
            'nesterov': False, 
            'maximize': False, 
            'foreach': None, 
            'differentiable': False, 
            'params': [0, 1, 2, 3, 4, 5]
        }
    ]
}
>>> optimizer.param_groups
[
    {
        'params': [
            Parameter containing:
                tensor([[-0.0604, -0.0189, -0.0246,  ..., -0.0696, -0.0631, -0.0584],
                        [ 0.0375, -0.0487, -0.0492,  ..., -0.0338,  0.0757,  0.0417],
                        [ 0.0306, -0.0683,  0.0532,  ...,  0.0034, -0.0688,  0.0165],
                        ...,
                        [ 0.0713,  0.0111, -0.0294,  ...,  0.0190,  0.0720,  0.0252],
                        [-0.0379, -0.0658,  0.0032,  ...,  0.0228, -0.0583, -0.0182],
                        [-0.0388,  0.0236,  0.0151,  ...,  0.0198, -0.0018, -0.0701]],
                    requires_grad=True), 
            Parameter containing:
                tensor([ 0.0508,  0.0297, -0.0020,  ..., -0.0418,  0.0486, -0.0348],
                    requires_grad=True), 
            Parameter containing:
                tensor([[[[ 0.1512,  0.0394,  0.0016],
                        [-0.1882,  0.0423,  0.1792],
                        [-0.1246,  0.1424,  0.0122]]]], requires_grad=True), 
            Parameter containing:
                tensor([0.0408], requires_grad=True), 
            Parameter containing:
                tensor([[[[ 0.0648,  0.2333,  0.0956],
                        [-0.2335,  0.1511,  0.1537],
                        [-0.0924, -0.0074,  0.1464]]]], requires_grad=True), 
            Parameter containing:
                tensor([0.1396], requires_grad=True)
        ], 
        'lr': 0.001, 
        'momentum': 0.9, 
        'dampening': 0, 
        'weight_decay': 0, 
        'nesterov': False, 
        'maximize': False, 
        'foreach': None, 
        'differentiable': False
    }
]
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
# 联合优化，nn.parameter.Parameter 的列表来连接不同模型的参数
grad_vars = list(model.parameters())
grad_vars += list(model_fine.parameters())

optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
```


```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

optimizer = torch.optim.SGD(params, lr=0.01)
```