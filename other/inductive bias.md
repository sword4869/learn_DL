归纳偏好。

An inductive bias is a set of assumptions that a learning algorithm uses to make predictions for new inputs that it has not encountered before. It is a way of guiding the learning process towards a preferred solution, based on some prior knowledge or belief. For example, if you want to learn a function that maps inputs to outputs, you might assume that the function is smooth and continuous, rather than jagged and discontinuous. This is an inductive bias that favors simpler and more generalizable functions over more complex and specific ones.

Inductive bias is essential for machine learning, because without it, there would be no way to choose among the infinitely many possible hypotheses that are consistent with the observed data. However, different learning algorithms may have different inductive biases, and some may be more suitable for certain problems than others. Therefore, it is important to understand the inductive bias of a learning algorithm and how it affects its performance and generalization ability.

Some examples of common inductive biases in machine learning are:

- Maximum conditional independence: 
    this is the assumption that the output variables are conditionally independent given the input variables. This is the bias used by the Naive Bayes classifier, which simplifies the computation of the joint probability distribution by ignoring the dependencies among the outputs.
- Minimum cross-validation error: 
    this is the criterion that selects the hypothesis with the lowest error on a validation set, which is a subset of the training data that is held out for evaluation. This is a way of avoiding overfitting, which occurs when the hypothesis fits the training data too well but fails to generalize to new data.
- Maximum margin: 
    this is the principle that tries to find a boundary between two classes that has the largest distance to the nearest points from both classes. This is the bias used by support vector machines, which aim to create a robust classifier that is less sensitive to noise and outliers.
- Minimum description length: 
    this is the idea that the best hypothesis is the one that compresses the data the most, by using fewer bits to encode it. This is based on Occam’s razor, which states that simpler explanations are more likely to be true than complex ones.
- Minimum features: 
    this is the strategy that eliminates irrelevant or redundant features from the input data, by using some measure of feature importance or selection. This can reduce the dimensionality and complexity of the problem, and improve the efficiency and accuracy of the learning algorithm.
- Nearest neighbors: 
    this is the assumption that points that are close to each other in the input space are likely to have similar outputs. This is the bias used by the k-nearest neighbors algorithm, which predicts the output of a new point by looking at its k closest neighbors in the training data and taking a majority vote or a weighted average.


CNN的 inductive bias 应该是locality和spatial invariance，即空间相近的grid elements有联系而远的没有，和空间不变性（kernel权重共享）RNN的inductive bias是sequentiality和time invariance，即序列顺序上的timesteps有联系，和时间变换的不变性（rnn权重共享）

CNN的平移不变性和局部性就是Inductive bias，它们是图像处理的特点吗，太是了！正因为这个原因，CNN在处理全局特征不够完美(不讨论CNN通过多层卷积或者空洞卷积也能提取全局特征)，于是Vision Transformer的self attention才能大行其道，

ViT的Inductive Bias就比较少，没有locality，效果也非常好，就是算力消耗多；Swin Transformer说你ViT的确牛，但是CNN里还有些好东西的，也没必要都扔掉，又捡了点回来，滑动窗口就是locality的体现，一些合适的Inductive Bias能减少很多计算量，有算力的就可以少一点 inductive bias 。

RNN假设当前的输入和之前的输入有关系，那么和之后的有关系吗？于是后来有了双向LSTM和BERT。
