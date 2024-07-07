- [1. install](#1-install)
- [2. distributed](#2-distributed)
- [3. for debugging](#3-for-debugging)
  - [3.1. fast run](#31-fast-run)
  - [3.2. Shorten the epoch length](#32-shorten-the-epoch-length)
- [4. Print input output layer dimensions](#4-print-input-output-layer-dimensions)
- [5. model weights summary](#5-model-weights-summary)
  - [5.1. call fit](#51-call-fit)
  - [5.2. 不 call fit](#52-不-call-fit)
- [6. checkpoint](#6-checkpoint)
  - [6.1. 自动开启](#61-自动开启)
  - [6.2. dir](#62-dir)
  - [6.3. 不存超参](#63-不存超参)
  - [6.4. 存超参](#64-存超参)
  - [6.5. resume for inference](#65-resume-for-inference)
  - [6.6. resume full training](#66-resume-full-training)
- [7. train](#7-train)
- [8. callbacks](#8-callbacks)
  - [8.1. eary stopping](#81-eary-stopping)
- [9. track](#9-track)

---

## 1. install

```python
pip install lightning
```

```python
# 1.x
import pytorch_lightning as pl

# 2.0
+ import lightning.pytorch as pl
```

## 2. distributed

Using `DistributedSampler` with the dataloaders. During `trainer.test()`, it is recommended to use `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once. Otherwise, multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have same batch size in case of uneven inputs.

It is recommended to use `self.log('val_loss', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.

## 3. for debugging
### 3.1. fast run
This argument will disable tuner, checkpoint callbacks, early stopping callbacks, loggers and logger callbacks like LearningRateMonitor and DeviceStatsMonitor.
```python
# runs 1 batch of training, validation, test and prediction
Trainer(fast_dev_run=True)
# runs 7
Trainer(fast_dev_run=7)
```

### 3.2. Shorten the epoch length
- `max_epochs=1`: 1 epoch
- `limit_train_batches`, `limit_val_batches`, `limit_test_batches`

    `limit_train_batches=100`: 1 个 epoch 内只有 100 个batch
  
    `limit_train_batches=0.1`: use only 10% of training data

## 4. Print input output layer dimensions

## 5. model weights summary

### 5.1. call fit

Whenever the `trainer.fit(...)` function gets called, the Trainer will print the weights summary for the LightningModule.

```
  | Name    | Type    | Params
------------------------------------
0 | encoder | Encoder | 50.4 K
1 | decoder | Decoder | 51.2 K
------------------------------------
```
To turn off the autosummary use:
```python
Trainer(enable_model_summary=False)
```

默认的summary只展示一层深度(相当于 `max_depth=1`)，如果要显示子模块，则

```python
from lightning.pytorch.callbacks import ModelSummary

trainer = Trainer(callbacks=[ModelSummary(max_depth=-1)])
```
```
  | Name         | Type       | Params
--------------------------------------------
0 | encoder      | Encoder    | 50.4 K
1 | encoder.l1   | Sequential | 50.4 K
2 | encoder.l1.0 | Linear     | 50.2 K
3 | encoder.l1.1 | ReLU       | 0     
4 | encoder.l1.2 | Linear     | 195   
5 | decoder      | Decoder    | 51.2 K
6 | decoder.l1   | Sequential | 51.2 K
7 | decoder.l1.0 | Linear     | 256   
8 | decoder.l1.1 | ReLU       | 0     
9 | decoder.l1.2 | Linear     | 51.0 K
--------------------------------------------
```

### 5.2. 不 call fit

```python

```

## 6. checkpoint

### 6.1. 自动开启

```python
# simply by using the Trainer you get automatic checkpointing
trainer = Trainer()


# disable
trainer = Trainer(enable_checkpointing=False)
```

### 6.2. dir
```python
trainer = Trainer()
'''
lightning_logs/
├── version_0
│   ├── checkpoints
│   │   └── epoch=0-step=100.ckpt
│   ├── events.out.tfevents.1695819024.eleven.971561.0
│   ├── events.out.tfevents.1695819046.eleven.971561.1
│   └── hparams.yaml
├── version_1       # 自动递增
'''

# saves checkpoints to 'some/path/' at every epoch end
trainer = Trainer(default_root_dir="log")
'''
log
└── lightning_logs
    ├── version_0
    │   ├── checkpoints
    │   │   └── epoch=0-step=100.ckpt
    │   ├── events.out.tfevents.1695821002.eleven.1007076.0
    │   ├── events.out.tfevents.1695821065.eleven.1007076.1
    │   └── hparams.yaml
'''
```
### 6.3. 不存超参

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate):


ckpt = torch.load(checkpoint_path)
print(ckpt)
'''
{
    "epoch": 0,
    "global_step": 100,
    "pytorch-lightning_version": "2.0.9",
    "state_dict": OrderedDict([...]),       # 网络的state_dict
    "loops": {
        "fit_loop": {...},
        "validate_loop": {...},
        "test_loop": {...},
        "predict_loop": {...}
    },
    "callbacks": {
        "ModelCheckpoint{...: None}": {...}
    },
    "optimizer_states": [      # optimizer的state_dict, 注意是序列
        {...}
    ],
    "lr_schedulers": []
}
'''
```

### 6.4. 存超参

`self.save_hyperparameters()`

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()     # 与 self.learning_rate = learning_rate 的前后位置无关---
'''
  {
      ...,
+     "hparams_name": "kwargs", 
+     "hyper_parameters": {"learning_rate": 0.001}}
  }
'''
# hyper_parameters 和 lightning_logs/version_1/hparams.yaml 的内容一样
```

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate, *args, **kwargs):

autoencoder = LitAutoEncoder(1e-3, 4, batch_size=32)

# 只存kwargs
'''
batch_size: 32
learning_rate: 0.001
'''
```

### 6.5. resume for inference

一般用于测试, `load_from_checkpoint` 只恢复 weight and hyperparameters of LightningModule
```python
- autoencoder = LitAutoEncoder(learning_rate=1e-3)
+ autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint_path)

print(autoencoder.learning_rate)
autoencoder.eval()
x = torch.randn(1, 64)

with torch.no_grad():
    y_hat = model(x)
```

其他的代码，从pl的ckpt中恢复
- optimizer
    ```python
    optimizer = autoencoder.configure_optimizers()
    optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    ```
- model
    
    ```python
    - class LitAutoEncoder(pl.LightningModule):
    -     def __init__(self, learning_rate):
    -         super().__init__()
    -         self.save_hyperparameters()
    -         self.encoder = Encoder()
    -         self.decoder = Decoder()

    + class AudoEncoder(nn.Module):
    +     def __init__(self):
    +         super().__init__()
    +         self.encoder = Encoder()
    +         self.decoder = Decoder()
    +     def forward(self, x):
    +         z = self.encoder(x)
    +         return self.decoder(z)
        
    + autoencoder2 = AudoEncoder().cuda()
    + autoencoder2.load_state_dict(checkpoint['state_dict'])
    ```
    ```python
    class Encoder(nn.Module):
        ...


    class Decoder(nn.Module):
        ...


    class Autoencoder(pl.LightningModule):
        def __init__(self, encoder, decoder, *args, **kwargs):
            ...


    autoencoder = Autoencoder(Encoder(), Decoder())
    checkpoint = torch.load(CKPT_PATH)
    encoder_weights = checkpoint["encoder"]
    decoder_weights = checkpoint["decoder"]
    ```

### 6.6. resume full training

model, optimizer, lr_schedulers 等等
```python
trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
```


## 7. train

If you return -1 here, you will skip training for the rest of the current epoch.
```python
class Autoencoder(pl.LightningModule):
    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx == 10:
            print("Training ends")
            return -1
'''
Epoch 0:  10%|███████████████▌                                                                                                                                           | 10/100 [00:00<00:01, 51.21it/s, v_num=8]Training ends
Epoch 1:  10%|███████████████▌                                                                                                                                           | 10/100 [00:00<00:02, 37.81it/s, v_num=8] 
'''
```

## 8. callbacks

### 8.1. eary stopping
```python
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class LitModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        loss = ...
        self.log("val_loss", loss)

early_stopping = EarlyStopping(monitor="val_loss")
trainer = Trainer(callbacks=[early_stopping])
```

常用
- `mode='min'`: `min`, `max`
- `patience=3`
    
    由于`check_on_train_epoch_end=False`则看val的次数, 所以还可以配合 val 的设置``check_val_every_n_epoch`` and ``val_check_interval``. i.e. with parameters ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training epochs before being stopped.

- `min_delta=0`: 绝对值. 默认`0`其实是一个很小的数。

有点意思
- `check_finite=True`: 防Nan和无限大
- `check_on_train_epoch_end=False`: 默认看val

没管：
- `stopping_threshold`
- `divergence_threshold`


## 9. track

```python
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        value = ...
        self.log("some_value", value)

values = {"loss": loss, "acc": acc, "metric_n": metric_n}  # add more items if needed
self.log_dict(values)
```
```python
self.log(..., prog_bar=True)
```
```python
# "min", "max", "mean"(default；avg也是这个), "sum"
self.log(..., reduce_fx="mean")
```
位置也由`default_root_dir`管理
```python
Trainer(default_root_dir="/your/custom/path")
```