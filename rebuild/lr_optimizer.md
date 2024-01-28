- [1. optimizer](#1-optimizer)
  - [1.1. torch.optim](#11-torchoptim)
  - [1.2. optimizer要优化的参数](#12-optimizer要优化的参数)
  - [1.3. optimizer报错](#13-optimizer报错)
- [2. lr\_scheduler](#2-lr_scheduler)
- [3. weight\_decay](#3-weight_decay)
- [4. 联合优化](#4-联合优化)


---

## 1. optimizer
### 1.1. torch.optim
```python
torch.optim.SGD(model.parameters(), lr=lr)

# Adam优化器的主要吸引力在于它对初始学习率不那么敏感。
torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
```

### 1.2. optimizer要优化的参数

要优化的参数：

- 可以传入 `model.parameters()`
    ```python
    torch.optim.SGD(model.parameters(), lr=lr)
    ```
- 也可以传入 由 `nn.Parameter` 

    ```python
    self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    self._scaling = nn.Parameter(scales.requires_grad_(True))
    rots = torch.zeros((N, 4), device="cuda")
    self._rotation = nn.Parameter(rots.requires_grad_(True))
    # self._opacity = nn.Parameter(opacities.requires_grad_(True))
    self._opacity = nn.Parameter(opacities.requires_grad_(False))

    l = [
        # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    ]

    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
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

### 1.3. optimizer报错
- `loss.backward()`: 计算model参数的grad. 
    
    梯度可累加，gradient_accumulation_steps
- `optimizer.step()`: optimizer更新自己的state, 还根据model参数的grad更新model参数的data
- `optimizer.zero_grad()`：model参数的grad置None
- `model.load_state_dict()`: 只恢复model的data，而model参数的grad还是None
    
    调整model参数顺序没影响，其是通过参数的名字来对应的。

- `optimizer.load_state_dict()`: optimizer恢复自己的state，还恢复其param_groups（包括lr、weight_decay等）

    optimizer的state是对应model的参数，shape一致。但一一对应不是通过参数的名字，而是`0,1,2`的顺序。所以当model的参数顺序变化时，其对应错误。
    ```
    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
    ```

<details>
<summary> 调整参数顺序，Model和Model2，optimizer训练时报错 </summary>

```python
from ast import mod
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate.utils import set_seed

set_seed(42)

class RangeDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.arange(length * 2).reshape(-1,2).float()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 4)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        return output
    
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = nn.Linear(3, 4)
        self.linear1 = nn.Linear(2, 3)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        return output

def show(optimizer, model):
    print(optimizer.state_dict()['state'])  # 有值
    print(dict(model.named_parameters()))
    print(model.linear1.weight.data) 
    print(model.linear1.weight.grad) 
    print('-'*10)

def train(rank):
    model = Model().to(rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    for i, data in enumerate(rand_dataloader):
        inputs = data.to(rank)
        outputs = model(inputs)
        labels = torch.randn(outputs.shape).to(rank)
        loss = F.mse_loss(outputs, labels)

        show(optimizer, model)
        loss.backward()
        show(optimizer, model)
        optimizer.step()
        show(optimizer, model)
        optimizer.zero_grad()
        show(optimizer, model)
        print('*'*10)
        if i == 1:
            break
    torch.save(model.state_dict(), f'./model_{rank}.pth')
    torch.save(optimizer.state_dict(), f'./optimizer_{rank}.pth')


def train2(rank):
    model = Model2().to(rank)
    ckpt = torch.load("model_0.pth", map_location="cpu")
    model.load_state_dict(ckpt)

    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    show(optimizer, model)
    ckpt = torch.load("optimizer_0.pth", map_location="cpu")
    optimizer.load_state_dict(ckpt)
    show(optimizer, model)

    for data in tqdm(rand_dataloader):
        inputs = data.to(rank)
        outputs = model(inputs)
        labels = torch.randn(outputs.shape).to(rank)
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        show(optimizer, model)
        optimizer.zero_grad()
rand_dataset = RangeDataset(length=100)
rand_dataloader = DataLoader(rand_dataset, batch_size=5)
train(0)
print('!'*20)
train2(0)
```
</details>

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