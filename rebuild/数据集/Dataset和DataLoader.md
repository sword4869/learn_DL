- [1. 数据集来源](#1-数据集来源)
  - [1.1. 官方数据集](#11-官方数据集)
  - [1.2. 自定义数据集](#12-自定义数据集)


---

`torchvision.datasets`(官方数据集) 返回 `torch.utils.data.Dataset`(Dataset) 对象.

`torch.utils.data.DataLoader`(DataLoader)加载`torch.utils.data.Dataset`(Dataset) 对象.

将DataSet 传递给DataLoader，DataLoader将DataSet根据batch_size分成几份，将然后通过DataLoader每次迭代。

