

---

`torchvision.datasets`(官方数据集) 返回 `torch.utils.data.Dataset`(Dataset) 对象.

`torch.utils.data.DataLoader`(DataLoader)加载`torch.utils.data.Dataset`(Dataset) 对象.

将DataSet 传递给DataLoader，DataLoader将DataSet根据batch_size分成几份，将然后通过DataLoader每次迭代。

---

```python
torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=None,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    *,
    prefetch_factor=None,
    persistent_workers=False,
    pin_memory_device="",
)
```
map-style:
DataLoader by default constructs a **index sampler** that yields **integral indices**. To make it work with a map-style dataset with **non-integral indices/keys**, **a custom sampler** must be provided.


`shuffle`是在每个**epoch**打乱，而不是每次迭代。

DataLoader 将根据 `shuffle` 参数自动构建 sequential or shuffled sampler。
使用`sampler`参数来自定义 sampler.

A custom Sampler that yields a list of batch indices at a time can be passed as the batch_sampler argument. Automatic batching can also be enabled via batch_size and drop_last arguments.

`sampler` 和 `batch_sampler` 都不兼容可迭代风格的数据集，因为这些数据集没有键或索引的概念。


`batch_sampler`:
- `None`(default)

`batch_size`
- `None`: 单个样本 (C,H,W)
- `1`(default): （1,C,H,W)

When both `batch_size` and `batch_sampler`(default) are None, automatic batching is disabled. When automatic batching is disabled, the default `collate_fn` simply converts NumPy arrays into PyTorch Tensors, and keeps everything else untouched.

When automatic batching is enabled, `collate_fn` is called with a list of data samples at each time. It is expected to collate the input samples into a batch for yielding from the data loader iterator.

For map-style datasets, users can alternatively specify batch_sampler, which yields a list of keys at a time.

The `batch_size` and `drop_last` arguments essentially are used to construct a `batch_sampler` from `sampler`. 
- For map-style datasets, the `sampler` is either provided by user or constructed based on the `shuffle` argument. 
- For iterable-style datasets, the `sampler` is a dummy infinite one


互斥：
- `sampler`; `shuffle`
- `batch_sampler`; `batch_size`, `shuffle`, `sampler`, `drop_last`.



> `collate_fn`: merges a list of samples(using the indices from sampler) to form a mini-batch of Tensor(s).

从数据集中获得的每个样本都作为参数传递给`collegel_fn`函数进行处理。

loading from a map-style dataset is roughly equivalent with:
```python
for indices in batch_sampler:
	yield collate_fn([dataset[i] for i in indices])
```

loading from an iterable-style dataset is roughly equivalent with:

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
	yield collate_fn([next(dataset_iter) for _ in indices])
```

```python

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class RangeDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.arange(0, length, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    
def collate_fn(batch):
    '''
    batch是从Dataset里直接获取的 batch_size 个元素的 list
    我们在 for 循环里对其每个元素作出处理后，再用 torch.stack 将 list 转化为 (N, shape) 的形式。
    比如, N个CHW 变成 NCHW
    '''
    batch = torch.stack([item + 1 for item in batch])
    return batch

# DataLoader
rand_dataset = RangeDataset(length=20)
rand_dataloader = DataLoader(
    rand_dataset, 
    batch_size=5,
    collate_fn=collate_fn
)

for batch in rand_dataloader:
    print(batch)
'''
tensor([1., 2., 3., 4., 5.])
tensor([ 6.,  7.,  8.,  9., 10.])
tensor([11., 12., 13., 14., 15.])
tensor([16., 17., 18., 19., 20.])
'''
```
```python
def collate_fn(batch):
    '''
    Dataset元素是字典
    '''
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in batch])
    embedding = torch.stack([example["embedding"] for example in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids, "embedding": embedding} 
```


`num_workers`:
- **Single-process data loading** (default `0`, 只在main process加载), 
  	
	**Multi-process data loading**(`num_workers`取正整数时, 会创建`num_workers`个 worker process)

	问题，1个main process 和 1 个 workder process, 那么效率一样不一样？
- 消耗CPU内存。overall memory usage is `num_workers` * `size of parent process`
- 具体创建多进程的时间：each time an iterator of a DataLoader is created, 比如调用`enumerate(dataloader)`，都会创建 `num_workers` 个 worker processes
- 具体销毁多进程时间：When `persistent_workers=False`, Workers are shut down once the end of the iteration is reached, or when the iterator becomes garbage collected. When `persistent_workers=True`, the dataloader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive.
- 平台：底层是基于python的`multiprocessing`[包](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing)而实现的，linux用`fork()`，windows和macos用`spawn()`。

	windows平台需要做额外的操作来满足 multi-process data loading：
    - `DataLoader`用`if __name__ == '__main__':`包裹
  		
		Wrap most of you main script's code within `if __name__ == '__main__':` block, to make sure it doesn't run again (most likely generating error) when each worker process is launched. You can place your dataset and [`DataLoader`](https://pytorch.org/docs/stable/data.html#iterable-style-datasets#torch.utils.data.DataLoader "torch.utils.data.DataLoader") instance creation logic here, as it doesn't need to be re-executed in workers.
    
	- `dataset`不在`if __name__ == '__main__':`中
    	
		Make sure that any custom `collate_fn`, `worker_init_fn` or `dataset` code is declared as top level definitions, outside of the `__main__` check. This ensures that they are available in worker processes. (this is needed since functions are pickled as references only, not `bytecode`.)

`drop_last`:
in multi-process loading, the `drop_last` argument drops the last non-full batch of each worker's iterable-style dataset replica.

`pin_memory=True`:
不建议在 multi-process loading 中直接返回CUDA张量。相反，我们建议使用自动内存固定（即设置`pin_memory=True`），在返回张量之前，DataLoader 将张量复制到设备/CUDA固定的内存中，从而能够更快地将数据传输到GPU。
pinned memory is page-locked memory.


---


`torch.utils.data.Dataset`
- map-style dataset
- overwrite `__len__()`, `__getitem__()`

`torch.utils.data.IterableDataset`
- iterable-style dataset
- overwrite `__iter__()`
- multi-process data loading 需要处理来防止重复。
  
	When a subclass is used with `DataLoader`, each item in the dataset will be yielded from the `DataLoader` iterator. When `num_workers > 0`, each worker process will have a different copy of the dataset object, so it is often desired to configure each copy independently to avoid having duplicate data returned from the workers. `get_worker_info()`, when called in a worker process, returns information about the worker. It can be used in either the dataset's `__iter__()` method or the `DataLoader` 's `worker_init_fn` option to modify each copy's behavior.

	Example 1: splitting workload across all workers in `__iter__()`:

	```
	>>> class MyIterableDataset(torch.utils.data.IterableDataset):
	...     def __init__(self, start, end):
	...         super(MyIterableDataset).__init__()
	...         assert end > start, "this example code only works with end >= start"
	...         self.start = start
	...         self.end = end
	...
	...     def __iter__(self):
	...         worker_info = torch.utils.data.get_worker_info()
	...         if worker_info is None:  # single-process data loading, return the full iterator
	...             iter_start = self.start
	...             iter_end = self.end
	...         else:  # in a worker process
	...             # split workload
	...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
	...             worker_id = worker_info.id
	...             iter_start = self.start + worker_id * per_worker
	...             iter_end = min(iter_start + per_worker, self.end)
	...         return iter(range(iter_start, iter_end))
	...
	>>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
	>>> ds = MyIterableDataset(start=3, end=7)

	>>> # Single-process loading
	>>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
	[tensor([3]), tensor([4]), tensor([5]), tensor([6])]

	>>> # Mult-process loading with two worker processes
	>>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
	>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
	[tensor([3]), tensor([5]), tensor([4]), tensor([6])]

	>>> # With even more workers
	>>> print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
	[tensor([3]), tensor([5]), tensor([4]), tensor([6])]

	```

	Copy to clipboard

	Example 2: splitting workload across all workers using `worker_init_fn`:

	```
	>>> class MyIterableDataset(torch.utils.data.IterableDataset):
	...     def __init__(self, start, end):
	...         super(MyIterableDataset).__init__()
	...         assert end > start, "this example code only works with end >= start"
	...         self.start = start
	...         self.end = end
	...
	...     def __iter__(self):
	...         return iter(range(self.start, self.end))
	...
	>>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
	>>> ds = MyIterableDataset(start=3, end=7)

	>>> # Single-process loading
	>>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
	[3, 4, 5, 6]
	>>>
	>>> # Directly doing multi-process loading yields duplicate data
	>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
	[3, 3, 4, 4, 5, 5, 6, 6]

	>>> # Define a `worker_init_fn` that configures each dataset copy differently
	>>> def worker_init_fn(worker_id):
	...     worker_info = torch.utils.data.get_worker_info()
	...     dataset = worker_info.dataset  # the dataset copy in this worker process
	...     overall_start = dataset.start
	...     overall_end = dataset.end
	...     # configure the dataset to only process the split workload
	...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
	...     worker_id = worker_info.id
	...     dataset.start = overall_start + worker_id * per_worker
	...     dataset.end = min(dataset.start + per_worker, overall_end)
	...

	>>> # Mult-process loading with the custom `worker_init_fn`
	>>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
	>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
	[3, 5, 4, 6]

	>>> # With even more workers
	>>> print(list(torch.utils.data.DataLoader(ds, num_workers=12, worker_init_fn=worker_init_fn)))
	[3, 4, 5, 6]
	```

`torch.utils.data.TensorDataset`

`torch.utils.data.ConcatDataset`

`torch.utils.data.ChainDataset`

`torch.utils.data.Subset`


---

`torch.utils.data.Sampler`

`torch.utils.data.SequentialSampler`

`torch.utils.data.RandomSampler`

`torch.utils.data.SubsetRandomSampler`

`torch.utils.data.WeightedRandomSampler`

`torch.utils.data.BatchSampler`

`torch.utils.data.distributed.DistributedSampler`
- 默认`shuffle=True`

- `set_epoch()`
  
  	In distributed mode, calling the `set_epoch()` method at the beginning of each epoch **before** creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.

	```python
	sampler = DistributedSampler(dataset, shuffle=True)	# 这里的shuffle只是第一个epoch要shuffle
	loader = DataLoader(dataset, sampler=sampler)
	for epoch in range(start_epoch, n_epochs):
		sampler.set_epoch(epoch)	# 让每次epoch都重新shuffle，没有这句则每次epoch都用第一个epoch的shuffle结果
		train(loader)
	```