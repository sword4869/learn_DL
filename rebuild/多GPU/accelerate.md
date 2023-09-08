- [1. Supported integrations](#1-supported-integrations)
- [2. migrate code](#2-migrate-code)
  - [2.1. basic](#21-basic)
  - [2.2. device](#22-device)
  - [2.3. Accelerator](#23-accelerator)
  - [2.4. checkpoints](#24-checkpoints)
    - [用于 training 的保存和恢复](#用于-training-的保存和恢复)
    - [同 torch.save](#同-torchsave)
  - [2.5. logger](#25-logger)
- [3. launch code](#3-launch-code)
  - [3.1. config](#31-config)
  - [3.2. CLI](#32-cli)
  - [3.3. notebook](#33-notebook)


---

<https://huggingface.co/docs/accelerate/index>

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

1. 可以删除**对模型和输入数据的调用**`.to(device)`或`.cuda()`。加速器对象将将其放置在适合您的设备上。当然可以继续保留。
2. **但是其他对象还需要**`.to(device)`，但你应该使用加速器对象提供的设备：accelerator.device。
    ```python
    # - device = 'cuda'
    device = accelerator.device
    ```

### 2.3. Accelerator

```python
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)
```

`gradient_accumulation_steps`: [OOM](./OOM.md#gradient_accumulation_steps)


```python
if accelerator.is_local_main_process:

if accelerator.is_main_process:
```

### 2.4. checkpoints

#### 用于 training 的保存和恢复

存`accelerator.prepare(...)`的一堆东西，由于其是自定义内容和顺序的，所以是 saving/loading everything.

从而，

```python
accelerator.save_state(f"checkpoint-{global_step}")
accelerator.load_state(f"checkpoint-{global_step}")
```

> 会保存随机状态 `random_states_0.pkl`, 所以适合 train, 不适合 inference

```python
from accelerate import Accelerator
import torch

accelerator = Accelerator()
checkpoint_path = './check'
accelerator.print(torch.randn((1, 4)))
accelerator.print(torch.randn((1, 4)))
accelerator.save_state(checkpoint_path)     <<<

$ python t.py
tensor([[1.7642, 0.3760, 1.3258, 0.7158]])
$ python t.py
tensor([[-0.9458, -2.2758,  0.3105, -0.7734]])
```
`load_state()` 前的还是随机，`load_state()`后的就是 checkpoint 定死的随机状态。
```python
from accelerate import Accelerator
import torch

accelerator = Accelerator()
checkpoint_path = './check'
accelerator.print(torch.randn((1, 4)))
accelerator.load_state(checkpoint_path)     <<<
accelerator.print(torch.randn((1, 4)))
accelerator.print(torch.randn((1, 4)))

'''
$ python t.py
tensor([[ 0.3128,  0.5455, -0.3177,  0.1281]])
tensor([[-1.6668,  0.4297,  0.6978, -0.3600]])  # 不变
tensor([[ 0.0186, -0.3621,  0.4523,  0.7063]])  # 不变
$ python t.py
tensor([[-0.9458, -2.2758,  0.3105, -0.7734]])
tensor([[-1.6668,  0.4297,  0.6978, -0.3600]])  # 不变
tensor([[ 0.0186, -0.3621,  0.4523,  0.7063]])  # 不变
'''
```
#### 同 torch.save

accelerate没有load，用torch的load
```python
accelerator.save(my_model.state_dict(), 'aa.bin')
my_model = torch.load('aa.bin')
```
### 2.5. logger

```python
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

logger.info(accelerator.state, main_process_only=False)
```



## 3. launch code

PS：还可以像原来那样直接运行 `python xxx.py`

### 3.1. config
These configs are saved to a `default_config.yaml` file in your cache folder for Accelerate.

This cache folder is located at (with decreasing order of priority):

- The content of your environment variable `HF_HOME` suffixed with accelerate.
- If it does not exist, the content of your environment variable `XDG_CACHE_HOME` suffixed with huggingface/accelerate.
- If this does not exist either, the folder `~/.cache/huggingface/accelerate`.

To have multiple configurations, the flag `--config_file` can be passed to the accelerate launch command paired with the location of the custom yaml.
`accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...`

```bash
accelerate launch \
  --mixed_precision="fp16" \
  --num_machines=1 --num_processes=4 --multi_gpu\
  xxx.py


`--num_machines` was set to a value of `1`
`--num_processes` was set to a value of `2`
        More than one GPU was found, enabling multi-GPU training.
        If this was unintended please pass in `--num_processes=1`.
`--mixed_precision` was set to a value of `'no'`
`--dynamo_backend` was set to a value of `'no'`
```

### 3.2. CLI
```bash
# pass in additional arguments and parameters to your script afterwards like normal!
accelerate launch {script_name.py} --arg1 --arg2 ...
```
For example, here is how to use accelerate launch with a single GPU:

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...
```

### 3.3. notebook
Accelerate also provides a `notebook_launcher` function you can use in a notebook to launch a distributed training. This is especially useful for Colab or Kaggle notebooks with a TPU backend. Just define your training loop in a `training_function` then in your last cell, add:
```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```