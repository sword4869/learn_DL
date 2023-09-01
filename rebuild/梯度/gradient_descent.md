## gradient descent

虽然随机梯度简单有效，但它需要仔细调整模型超参数，特别是优化中使用的**学习率**以及**模型参数的初始值**。

因为由于每一层的输入都受到所有先前层的参数的影响，因此，随着网络变得更深，网络参数的微小变化会放大。

## Stochastic gradient descent (SGD)

SGD variants such as momentum (《On the importance of initialization and momentum in deep learning》 2013) and Adagrad (《Adaptive subgradient methods for online learning and stochastic optimization》 2011)


mini-batch GD 比 单个GD 好：
- 小批量损失的梯度是对训练集梯度的估计，其质量随着批量大小的增加而提高。批量大小越大，估计越靠谱。
- 由于GPU并行性，批量计算比单个样本的 m 次计算要高效得多。