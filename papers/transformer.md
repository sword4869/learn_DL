
## 输入特征

*Local Implicit Ray Function for Generalizable Radiance Field Representation* 中关于transformer的输入都不是直接输入concatenate起来的features的，而是concatenate后再经过MLP后的。这样起一个 **reduce feature channels** 的作用。

小MLP。The “MLP” is a two-layer MLP and the number of channels is set to 32.

![1689674494033057777.png](../images/1689674494033057777.png)
 
