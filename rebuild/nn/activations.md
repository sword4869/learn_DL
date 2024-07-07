- [1. Sigmoid, Tanh](#1-sigmoid-tanh)
  - [1.1. sigmoid](#11-sigmoid)
  - [1.2. Tanh](#12-tanh)
  - [1.3. Improvements](#13-improvements)
- [2. Rectified Activation Functions](#2-rectified-activation-functions)
  - [2.1. ReLU](#21-relu)
  - [2.2. LeakyReLU 2013](#22-leakyrelu-2013)
  - [2.3. SiLU 2018](#23-silu-2018)
- [3. Miscellaneous Activation Functions](#3-miscellaneous-activation-functions)
  - [3.1. softplus](#31-softplus)
  - [3.2. GELU 2016](#32-gelu-2016)
- [4. 参数inplace=True](#4-参数inplacetrue)

为什么需要激活函数？
1. 激活函数避免“层数的塌陷”。意思是如果不用激活函数，那么数层线性层完全就是一层线性层，而有了激活函数，加入了非线性后，上下层之间就不是线性关系了。

    the non-linearity needs to be introduced in the neural networks. Otherwise, a neural network produces the output as a linear function of inputs inspite of having several layers. 
2. Moreover, in practice data is generally not linearly separable; hence, the non-linear layers help to project the data in non-linear fashion in feature space

为什么最后一层不加激活函数？
因为最后一层不需要避免层数的塌陷，没有下一层。

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062009973.png)

[Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark. 2021](https://arxiv.org/pdf/2109.14545.pdf)

## 1. Sigmoid, Tanh
### 1.1. sigmoid
[0, 1]

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062009974.png)

`nn.Sigmoid`, 


$\mathbf{f}(x)=\dfrac{1}{1+\exp(-x)}$

> 特点: 

- squash input in [0, 1]带来问题:
  
  1. The output of the Logistic Sigmoid function is saturated for higher and lower inputs, which leads to **vanishing gradient problem** (The vanishing gradient problem depicts to a scenario where the gradient of objective function w.r.t. a parameter becomes very close to zero and leads to almost no update in the parameters during the training of the network using stochastic gradient descent technique). 

  2. Moreover, the output **not following a zero-centric** nature leads to poor convergence.

- optimization difficulty: 
  
  The training of deep networks become difficult due to **the uniform slope** of the Logistic Sigmoid and Tanh AFs **near the origin**.
- computational complexity

### 1.2. Tanh 

`nn.Tanh`, 

[-1, 1]

$\mathrm{f}(x)=\tanh(x)=\dfrac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}$

> 特点: 

- squash input in [−1, 1]. 
  
    **比 sigmoid 解决了zero-centric**，但是还存在vanishing gradient problem
- computational complexity


### 1.3. Improvements
- scaled Hyperbolic Tangent (sTanh). *Gradient-based learning applied to document recognition. 1998*
- Penalized Tanh (pTanh)
- A Parametric Sigmoid Function (PSF)
- Scaled Sigmoid (sSigmoid)
- improved logistic Sigmoid (ISigmoid). *The optimized deep belief networks with improved logistic sigmoid units and their application in fault diagnosis for planetary gearboxes of wind turbine. 2018*
- Elliott. *A comparative performance analysis of different activation functions in lstm networks for classification.2019*
- Linearly scaled hyperbolic tangent (LiSHT). *Lisht: Nonparametric linearly scaled hyperbolic tangent activation function for neural networks.2019*
- Soft-Root-Sign (SRS, *Soft-root-sign activation function.2020*

## 2. Rectified Activation Functions

### 2.1. ReLU
`nn.ReLU`, 

$[0, \infty]$

$\mathbf{f}(x)=\max(0,x)$ 

or 

$\mathbf{f}(x)=\begin{cases}x,&\text{if }x\geq0\\0,&\text{otherwise}\end{cases}$




> 特点:

- 解决computational complexity: ReLU相比sigmoid的指数运算很贵，而不需要指数运算的ReLU很快。
- 但还存在 vanishing gradient problem for the negative inputs.
  
  Non-utilization of Negative Values. The gradient for positive and negative inputs is one and zero.


### 2.2. LeakyReLU 2013

*Rectifier nonlinearities improve neural network acoustic models.2013*

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062009975.png)

`nn.LeakyReLU`, 

$\mathbf{f}(x)=\max(0,x) + \text{negative\_slope} ∗ \min(0,x)$ 

or 

$\mathbf{f}(x)=\begin{cases}x,&\text{if }x\geq0\\\text{negative\_slope}\times x,&\text{otherwise}\end{cases}$

$[-\infty, +\infty]$

### 2.3. SiLU 2018

*Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. 2018*


The SiLU outperforms the ReLU function for **reinforcement learning**.


![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062009976.png)

`nn.SiLU()`, 

$\mathbf{f}(x)=\dfrac{x}{1+\exp(-x)}$, 

(−0.5, ∞)

`x * nn.Sigmoid(x)`.

## 3. Miscellaneous Activation Functions
### 3.1. softplus

*Incorporating second-order functional knowledge for better option pricing.2001*

*Improving deep neural networks using softplus units. 2015*

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062009977.png)

`nn.Softplus`, 

$\mathbf{f}(x) = \log(1 + \exp\mathbf{x})$

> 特点

The smooth nature of the Softplus facilitates the differentiability.

### 3.2. GELU 2016
Gaussian Error Linear Unit, *Gaussian error linear units (gelus), arXiv preprint.2016*

$\mathbf{f}(x)= x∗\Phi(x)$, $\Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution.

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062009978.png)



## 4. 参数inplace=True

`nn.ReLU(inplace=True)`
- inplace：can optionally do the operation in-place. 
- 默认 `input=False`, 不会改变原输入，只会产生新的输出
- `inplace=True`，将会改变输入的数据, 可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。会对原变量覆盖，只要不带来错误就用。