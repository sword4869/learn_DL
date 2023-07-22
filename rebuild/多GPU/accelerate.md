- [1. Supported integrations](#1-supported-integrations)
- [2. migrate code](#2-migrate-code)
  - [2.1. basic](#21-basic)
  - [2.2. device](#22-device)
  - [2.3. checkpoints](#23-checkpoints)
- [3. launch code](#3-launch-code)
  - [3.1. CLI](#31-cli)
  - [3.2. notebook](#32-notebook)


---
This repository is tested on Python 3.7+ and PyTorch 1.4.0+
```python
pip install accelerate
```

## 1. Supported integrations

- CPU only
- multi-CPU on one node (machine)
- multi-CPU on several nodes (machines)
- single GPU
- multi-GPU on one node (machine)
- multi-GPU on several nodes (machines)
- TPU
- FP16 with native AMP (apex on the roadmap)
- DeepSpeed support (Experimental)
- PyTorch Fully Sharded Data Parallel (FSDP) support (Experimental)
- Megatron-LM support (Experimental)
## 2. migrate code 

### 2.1. basic
```python
from accelerate import Accelerator

accelerator = Accelerator()

# 顺序随便，个数随便
model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
)


for batch in training_dataloader:
    # -   inputs = inputs.to(device)
    # -   targets = targets.to(device)
    inputs, targets = batch
    outputs = model(inputs)

    loss = loss_function(outputs, targets)
    
    optimizer.zero_grad()
    # -   loss.backward()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

Then call `accelerator.prepare()` passing in the **PyTorch objects** that you would normally train with. This will return the same objects, but they will be on the correct device and distributed if needed. Also note that you don't need to call `model.to(device)` or `inputs.to(device)` anymore, as this is done automatically by `accelerator.prepare()`.
### 2.2. device

1. 可以删除对模型和输入数据的调用`.to(device)`或`.cuda()`。加速器对象将为您处理此问题，并将所有这些对象放置在适合您的设备上。
2. 或者，可以把这些`.to(device)`调用留给别人，但你应该使用加速器对象提供的设备：accelerator.device。
    ```python
    # - device = 'cuda'
    device = accelerator.device
    ```

### 2.3. checkpoints
存`accelerator.prepare(...)`的一堆东西，由于其是自定义内容和顺序的，所以是 saving/loading everything.

从而，

```python
accelerator.save_state("checkpoint_dir")
accelerator.load_state("checkpoint_dir")
```


## 3. launch code
### 3.1. CLI
```bash
# pass in additional arguments and parameters to your script afterwards like normal!
accelerate launch {script_name.py} --arg1 --arg2 ...
```
For example, here is how to use accelerate launch with a single GPU:

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...
```

### 3.2. notebook
Accelerate also provides a `notebook_launcher` function you can use in a notebook to launch a distributed training. This is especially useful for Colab or Kaggle notebooks with a TPU backend. Just define your training loop in a `training_function` then in your last cell, add:
```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```