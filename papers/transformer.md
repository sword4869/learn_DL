- [1. 目标](#1-目标)
- [2. 结构](#2-结构)
  - [2.1. scaled dot-product attention](#21-scaled-dot-product-attention)
    - [2.1.2. scaled dot-product attention 和 additive attention, dot-product attention 的区别](#212-scaled-dot-product-attention-和-additive-attention-dot-product-attention-的区别)
    - [2.1.3. mask](#213-mask)
  - [2.2. FC: Position-wise Feed-Forward Networks](#22-fc-position-wise-feed-forward-networks)
- [3. 输入特征](#3-输入特征)

---

《2017. Attention Is All You Need》

## 1. 目标

To the best of our knowledge, however, the Transformer is the first transduction model **relying entirely on self-attention** to compute representations of its input and output **without using sequencealigned RNNs or convolution**.

## 2. 结构

![Attention Is All You Need](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012994.png)  

![图 10](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012995.png)  


### 2.1. scaled dot-product attention

《Attention Is All You Need》中的attention。

#### 2.1.2. scaled dot-product attention 和 additive attention, dot-product attention 的区别

![Attention Is All You Need - Section 3.2.1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012996.png)  


![Attention Is All You Need - Section 3.2.1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012997.png)  


#### 2.1.3. mask


![Attention Is All You Need - Fig.2](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012998.png)  


第一步： query 和 key 相乘，然后缩放，得到权值 s

第二步：是否mask

第三步：将权值进行归一化，得到直接可用的权重 a

第四步：将权重 a 和 value 进行加权求和

![Attention Is All You Need - Section 3.2.3](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012999.png)  

softmax 是 $e^x$, 当 $x = -\infty$ 时， $e^x \approx 0$.


### 2.2. FC: Position-wise Feed-Forward Networks

![图 11](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012000.png)  


## 3. 输入特征

*Local Implicit Ray Function for Generalizable Radiance Field Representation* 中关于transformer的输入都不是直接输入concatenate起来的features的，而是concatenate后再经过MLP后的。这样起一个 **reduce feature channels** 的作用。

小MLP。The “MLP” is a two-layer MLP and the number of channels is set to 32.

![1689674494033057777.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012001.png)

