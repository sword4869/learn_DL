- deterministic 为 True 能保证复现性
- benchmark 为 True 不保证计算更快

## 1. benchmark
cuDNN 是英伟达专门为深度神经网络所开发出来的**GPU加速库**. 

PyTorch 会默认使用 cuDNN 加速。

```python
print(torch.backends.cudnn.enabled) # True
# torch.backends.cudnn.enabled = True 没必要写。
```

但是，torch.backends.cudnn.benchmark 默认模式是为 False
```python
torch.backends.cudnn.benchmark = True
```

benchmark 将会让程序在第一次迭代时花费更多时间，为整个网络的每个卷积层搜索最适合它的**卷积实现算法**(有的算法在卷积核大的情况下，速度很快；比如有的算法在某些情况下内存使用比较小)，进而实现网络的加速。

要求模型的输入大小不能变。对于一个卷积层，这次的输入形状比如是 (8, 3, 224, 224)，下次换成了 (8, 3, 112, 112)，那不行。对于一般的 CV 模型来说，网络的结构一般是不会动态变化的。

无关 batch size，即[不需要设置 drop_last in DataLoader](https://discuss.pytorch.org/t/if-i-set-torch-backends-cudnn-benchmark-true-should-i-also-set-drop-last-in-dataloader/137860)

保证最快，但不保证比不用**更快**。因为此flag是遍历执行所有卷积实现算法再找全局最优，而不用此flag时，pytorch会用启发式算法找快的卷积实现算法，可能这个局部最优就是全局最优。

## 2. deterministic

```python
torch.backends.cudnn.deterministic = True
```

将这个 flag 置为 True 的话，每次返回的卷积算法将是**确定的**，即默认算法（不设置的话，用启发式算法会找到不一样的实现算法）。

如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。

所以，设置 deterministic 为 True 时，benchmark 不设置。