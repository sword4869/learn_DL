![图 1](../images/057506b8b466816e419fb45541fd2beefb5f9ce2aaf49135be38f0af8be50c9e.png)  
![图 2](../images/db14ff16b503e6d17f64d1bef0c841e6e22d36ba1591be8c1af74fc4366004f8.png)  


- 有FC和其他类型（conv、self-attention...）的网络，FC的层叫做FC（而不是MLP）
- 全是FC的网络叫做MLP

> 请问多层感知机，前馈神经网络，深度神经网络，全连接神经网络这几个概念有什么区别？

最“大”的概念是人工神经网络（Artificial Neural Network，ANN）

两种性质：
- 前馈神经网络(Feed-Forward Network)
    
    此概念核心在于网络信息的传递是从输入到输出的单向过程，区别于后续的RNN等递归（循环）神经网络。比如，CNN, Transformer

- 全连接网络(Fully Connected Network)
    
    每层的每个节点都与其前后层的全部节点相连。

所以，多层感知器（MLP）既是前馈神经网络，又是全连接网络。