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

![Attention Is All You Need](../images/52cfaae3a70e0c41b21b6e755ec993babe73c24ee70408e77b2a5f32ccc099f9.png)  

![图 10](../images/452c8abd70ec99e77c64a9c716d845609a70e894bbef042bd2abb15d9e1ff868.png)  


### 2.1. scaled dot-product attention

《Attention Is All You Need》中的attention。

#### 2.1.2. scaled dot-product attention 和 additive attention, dot-product attention 的区别

![Attention Is All You Need - Section 3.2.1](../images/e6e13bb69156ae74326053a92d86c58a46c7f64e630cccd432bf98ff24b6cd59.png)  


![Attention Is All You Need - Section 3.2.1](../images/81c4db18b9792f1174c61abe97ee5823e6850b5703b402e697143af3643941dd.png)  


#### 2.1.3. mask


![Attention Is All You Need - Fig.2](../images/1aabed171f536e1da66cf090575508e6c864434342dfd92b92a5390d4959cf7d.png)  


第一步： query 和 key 相乘，然后缩放，得到权值 s

第二步：是否mask

第三步：将权值进行归一化，得到直接可用的权重 a

第四步：将权重 a 和 value 进行加权求和

![Attention Is All You Need - Section 3.2.3](../images/6194361e817193c87728a1748176113ccc609c8f50937c1bbf112a863455ac05.png)  

softmax 是 $e^x$, 当 $x = -\infty$ 时， $e^x \approx 0$.


### 2.2. FC: Position-wise Feed-Forward Networks

![图 11](../images/ed574320ebca9af246669d0471898de3c33193df11402443af7304e2c9fbce3b.png)  


## 3. 输入特征

*Local Implicit Ray Function for Generalizable Radiance Field Representation* 中关于transformer的输入都不是直接输入concatenate起来的features的，而是concatenate后再经过MLP后的。这样起一个 **reduce feature channels** 的作用。

小MLP。The “MLP” is a two-layer MLP and the number of channels is set to 32.

![1689674494033057777.png](../images/1689674494033057777.png)
 
