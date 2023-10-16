Sigmoid: [0, 1]

Tanh: [-1, 1]

ReLU: $[0, \infty]$

softplus: $\mathbf{f}(x) = \log(1 + e^ \mathbf{x})$

ReLU相比sigmoid的指数运算很贵，而不需要指数运算的ReLU很快。


为什么需要激活函数？
激活函数避免“层数的塌陷”，意思是如果不用激活函数，那么数层线性层完全就是一层线性层，而有了激活函数，加入了非线性后，上下层之间就不是线性关系了。

为什么最后一层不加激活函数？
因为最后一层不需要避免层数的塌陷，没有下一层。


## 参数inplace=True

`nn.ReLU(inplace=True)`
- inplace：can optionally do the operation in-place. 
- 默认 `input=False`, 不会改变原输入，只会产生新的输出
- `inplace=True`，将会改变输入的数据, 可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。会对原变量覆盖，只要不带来错误就用。