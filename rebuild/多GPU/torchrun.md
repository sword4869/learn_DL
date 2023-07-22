- [1. 多GPU](#1-多gpu)
  - [1.1. 单机多卡 single-node multi-GPU](#11-单机多卡-single-node-multi-gpu)
    - [1.1.1. DataParallel](#111-dataparallel)
  - [1.2. DistributedDataParallel](#12-distributeddataparallel)


---
# 1. 多GPU

- `multiprocessing`
  
  There are significant caveats to using CUDA models with `multiprocessing`; unless care is taken to meet the data handling requirements exactly, it is likely that your program will have incorrect or undefined behavior.

- `DataParallel`
  
  It is recommended to use `DistributedDataParallel`, instead of `DataParallel` to do multi-GPU training, **even if there is only a single node**.

## 1.1. 单机多卡 single-node multi-GPU

### 1.1.1. DataParallel


```python
from torch import nn
net = nn.Sequential(nn.Linear(3, 1)).cuda()
net = nn.DataParallel(net)
```

先`net.cuda()` 还是先 `nn.DataParallel(net)`都行, 但是必须有`net.cuda()`

`net.to(device)`也行, 送到哪个GPU上无所谓, 反正都会再复制到所有指定的GPU上.

> to(device)版本


```python
import torch
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size).to(device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)


for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

    Let's use 2 GPUs!
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    	In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
    

> cuda()版本


```python
import torch
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size).cuda()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)


for data in rand_loader:
    input = data.cuda()
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

    Let's use 2 GPUs!
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    	In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    	In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
    

## 1.2. DistributedDataParallel

- process group

  The devices to synchronize across are specified by the input process_group, **which is the entire world by default**.

  Creation of this class requires that torch.distributed to be already initialized, by calling `torch.distributed.init_process_group()`.

  This utility and multi-process distributed (single-node or multi-node) GPU training currently only achieves the best performance using the `NCCL` distributed backend. Thus NCCL backend is the recommended backend to use for GPU training.

- DistributedSampler
  
  DistributedDataParallel does not chunk or otherwise shard the input across participating GPUs; the user is responsible for defining how to do so, for example through the use of a `DistributedSampler`.

> launch

`torchrun` provides a superset of the functionality as `torch.distributed.launch` with the following additional functionalities:

- Worker failures are handled gracefully by restarting all workers.

- Worker `RANK` and `WORLD_SIZE` are assigned automatically.

- Number of nodes is allowed to change between minimum and maximum sizes (elasticity).

NOTE: `torchrun` is a python console script to the main module `torch.distributed.run` declared in the entry_points configuration in setup.py. It is equivalent to invoking `python -m torch.distributed.run`.

<https://pytorch.org/docs/stable/elastic/run.html#launcher-api>

```python
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


if __name__ == "__main__":

    torch.distributed.init_process_group(backend="nccl")
    device = torch.distributed.get_rank()
    print(f"Start running basic DDP example on device {device}.")

    # create model and move it to GPU with id device
    model = ToyModel().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()
```

关键:
- `torch.distributed.init_process_group(backend="nccl")`
- `device = torch.distributed.get_rank()`
- `model = ToyModel().to(device)`, `ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])`


```python
# --nproc_per_node=NUM_GPUS_YOU_HAVE
!torchrun --nproc_per_node=2 DDP-example.py

# or
!python -m torch.distributed.run --nproc_per_node=2 DDP-example.py
```

    WARNING:torch.distributed.run:
    *****************************************
    Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
    *****************************************
    Start running basic DDP example on rank 1.
    Start running basic DDP example on rank 0.
    
