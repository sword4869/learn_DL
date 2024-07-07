- [1. 退化现象](#1-退化现象)
  - [1.1. 什么是退化现象](#11-什么是退化现象)
  - [1.2. ResNet的误区](#12-resnet的误区)
- [2. res block](#2-res-block)
  - [2.1. one block](#21-one-block)
  - [2.2. multiple blocks](#22-multiple-blocks)
  - [2.3. 残差结构优化](#23-残差结构优化)
- [3. 理解](#3-理解)
  - [3.1. 恒等映射](#31-恒等映射)
  - [3.2. 逐步优化](#32-逐步优化)
  - [3.3. 综合](#33-综合)
  - [3.4. skip connection的两种区别](#34-skip-connection的两种区别)

---
## 1. 退化现象

### 1.1. 什么是退化现象

![168967463768607191161.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013552.png)

![16896746376870737898.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013553.png)

56层的网络无论是在训练集还是测试集上，误差率都比20层的要高。

出现这种现象的原因并非是由于层数加深引发的梯度消失/梯度爆炸，因为已经通过归一化的方法 BatchNorm 解决了这个问题。

我们将这种反常的现象称之为“退化现象”。

### 1.2. ResNet的误区

ResNet最直接的目的：**解决训练优化问题（退化问题）**，而**不是**一个直接提升模型效果的机制。

深层网络由于训练问题而连较好的局部最优都达不到，ResNet在优化到极值点的情况下，并不能比不带残差的模型更好。


## 2. res block

### 2.1. one block

![16896746376880736930.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013554.png)

一个残差块有2条路径 $F(x)$ 和 $x$ ，
- $F(x)$ 路径拟合残差，不妨称之为残差路径；
- $x$ 路径为 `identity mapping` 恒等映射，称之为`shortcut`。
- 图中的 $\oplus$ 为 **element-wise addition**，要求参与运算的 $F(x)$ 和 $x$ 的尺寸要相同.

### 2.2. multiple blocks

是新的$x_{l+k}$在跳，而不是$x_l$

![168967463768907211548.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013555.png)

![168967463769007258326.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013556.png)

![168967463769007258473.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013557.png)

### 2.3. 残差结构优化

![168967463769157814416.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013558.png)

上图是作者修改激活函数和BN层不同位置后得到的残差结构，这里最后一个效果最好。


![168967463769258431197.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013559.png)

## 3. 理解

解释的理论听听就行，听说Google也做了好多不同的结构来试验，结果ResNet这个成了。

**说白了肯定还有很多结构说着也有道理，但实验结果不咋地**。

![168967463769367939397.png](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013560.png)



### 3.1. 恒等映射

一个4层的MLP已经在训练集和测试集得到了100%的性能。又设立一个8层的MLP，copy了前四层的参数。如果这个新的网络也要达到性能100%，则新增加的层要做的事情就是“恒等映射”, $x+F(x)=x \to F(x)=0$.

退化现象表明了，实际上新增加的层，很难做到恒等映射。又或者能做到，但在有限的时间内很难完成（即网络要用指数级别的时间才能达到收敛）。

ResNET就是通过添加“桥梁”，将输入新层的数据直接送到新层的输出中，那么新层要做的事情就是把这些层的参数逼近于0，从而实现100%的性能。

### 3.2. 逐步优化


相当于直接去学一个复杂的波形，还是逐步优化一个粗略的波形到精确。

- 不用 shortcut

    信息误传。经过n层，每一层都是将前一层学到的信息再转录一遍。因为每层只关注前一层信息，那么如果一旦前一层误传了某个信息，后面的层都会跟着背离，从而到第n层时积累的误传已经太大了。

    信息缺失。前一层提取到信息中直接漏掉了某个关键信息，那么后续的层别说进一步提取更抽象的了，都不知道有这个信息。

- 用 shortcut

    前4层学到的大致准确的关键信息，element-wise addition 让残差不需要再用自己的话转录一边，而是只需在其上修修改改就行。

    好像前四层学到了低频信息，再 element-wise addition  加上新四层的高频信息，从而分工合作。

    ![图 8](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013561.png)  



### 3.3. 综合

自适应的网络：逐步优化，一旦前k层已经优化到最有结果了，那么后l层就会恒等映射，网络的深度太深就不是问题了。


### 3.4. skip connection的两种区别

skip connection:
- element-wise addition skip connections
  逐步优化 element-wise addition：整体简单的逐步优化到整体复杂。
  residual net
- concatenation skip connections
  分工合作 concatenate: 各局部相对简单的信息再拼出整体复杂
  Unet



