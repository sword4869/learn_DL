![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062018334.png)  
![图 2](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062018335.png)  


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