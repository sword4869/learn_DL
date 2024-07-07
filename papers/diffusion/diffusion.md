- [1. 里程碑](#1-里程碑)
- [2. 训练/加噪/扩散/forward/q](#2-训练加噪扩散forwardq)
  - [2.1. toy understanding](#21-toy-understanding)
  - [2.2. 实际上](#22-实际上)
- [3. 推理/生成/去噪/Sampling/p](#3-推理生成去噪samplingp)
  - [3.1. toy understanding](#31-toy-understanding)
  - [3.2. 实际上](#32-实际上)
  - [3.3. scheduler](#33-scheduler)
- [4. UNet](#4-unet)
- [5. DDPM](#5-ddpm)
- [6. DDIM](#6-ddim)
- [7. LDM](#7-ldm)



---

## 1. 里程碑

Diffusion Probabilistic Model (DPM) 简称 Diffusion Model

Denoising Diffusion Probabilistic Models (DDPM) ，多了个 Denoising.

Latent Diffusion Models (LDM)

- Diffusion models： 《Deep unsupervised learning using nonequilibrium thermodynamics》
- DDPM: 《Denoising Diffusion Probabilistic Models》
- latent diffusion & cross-attention: 《High-Resolution Image Synthesis with Latent Diffusion Models》
- DDIM: 《Denoising Diffusion Implicit Models》
- CFG(classifier free guidance): [Classifier-free diffusion guidance](https://arxiv.org/abs/2207.12598)
- sedit: 《Sdedit: Image synthesis and editing with stochastic differential equations》


![Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015109.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015110.png)

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>

---

## 2. 训练/加噪/扩散/forward/q

### 2.1. toy understanding

> 分开理解

当成supervised learning理解：
- 输入特征:  noisy image 和 其噪声等级 noise amount
- 让unet预测噪声 predicted noise
- 重建噪声：计算 predicted noise 和 ground truth noise (Label) 的 loss 来更新unet

noisy image = image + ground truth noise (Label) 

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015111.png)
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015112.png)
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015113.png)

> 整体理解

对一张训练图片，混合不同 noise amount 程度的 ground truth noise 得到 noisy image, noisy image 再通过 unet 得到 predicted noise, 计算 predicted noise 和 ground truth noise (Label) 的 loss 来更新unet.

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015114.png)

### 2.2. 实际上

> 迭代

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015115.png)

想象中，每一步的 noisy image 是 上一次迭代的 noisy image 加上噪音得到。

实际上，每一步的 noisy image 是直接在原始图片上加噪音得到。

> 只有一步

采用更稳定的训练策略：不是对同一张图片施加随时间步增长（遍历所有的时间步）的噪声等级，而是每次随机取一张图片，对其采**仅一次**随机的时间步的噪声等级。

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015116.png)

$$L=\mathbb{E}_{x,\epsilon\sim\mathcal{N}(0,1),t}\Big[\|\epsilon-\epsilon_{\theta}(x_{t},t)\|_{2}^{2}\Big]$$
- noise amount 即时间步 timestep。$t \in[1,2,3,...,T]$.
- $\epsilon$ 是 ground truth noise 
- $\epsilon_\theta$ 是unet, $\epsilon_\theta(x_t,t)$ 得到 predicted noise
- $x_t = \sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon$ : noisy image 由 原始图片 $x_0$ 和 噪声 $\epsilon$ 加权求和得到。

也有另一种写法，示意，训练还是如上：

$$\mathbb{E}_{t,\epsilon,x_0}[w(\lambda_t)\|x_0-\hat{x}_\theta(x_t,t)\|_2^2]$$

- $x_0$: 原始图片
- $x_t$: 加噪的图片
- $\hat{x}_\theta$: denotes the learned denoising model.
- $w(\lambda_t)$: 依赖于时间步长的加权常数

## 3. 推理/生成/去噪/Sampling/p

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015117.png)

经过 T 步 Denoise, 由噪声 $x_T$ 得到清晰的图片 $x_0$, 那么 Denoise具体是什么

### 3.1. toy understanding

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015118.png)

Noise sample 减去 神经网络预测出来的 Predicted Noise, 得到 更清晰的图像。 

`noise_sample = noise_sample - predicted_noise`, $x_{t-1} = x_t - \epsilon_\theta(x_{t}, t)$

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015119.png)

但这样的结果得到的只是乱七八糟的图片。

### 3.2. 实际上

- 再加入其他的噪音 $z$. 
- 还要让 ddpm scheduler 再预测出三个缩放因子，缩放 sample、predicted_noise、extra_noise。

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015120.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015121.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015122.png)

the very last step (i.e., t = 1), at which $\epsilon = 0, \sigma^2_t=\beta_t$.

- $s_1=\dfrac{1}{\sqrt{\alpha_t}}$
- $s_2=\dfrac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}$
- $s_3=\sigma_t$

### 3.3. scheduler

- $\beta_t$:
  
    is defined for every timestep $t$ accoridng to some schedule, and determines how much noise is added per timestep. 

- $\alpha_t = 1-\beta_t$
- 连乘 $\displaystyle \bar{\alpha_t} = \prod^T_{t=1}{\alpha_t} = \prod^T_{t=1}{1-\beta_t} = ({1-\beta_1})({1-\beta_2})...({1-\beta_t})$
  
```python
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt

scheduler = DDPMScheduler(num_train_timesteps=1000)
# scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
alphas = scheduler.alphas
betas = scheduler.betas
alphas_cumprod = scheduler.alphas_cumprod
sqrt_alphas_cumprod = scheduler.alphas_cumprod.sqrt()
sqrt_one_minus_alphas_cumprod = (1 - scheduler.alphas_cumprod) ** 0.5

plt.plot([1 for _ in scheduler.timesteps], label="no scaling")
plt.plot(betas, label=r"$\beta_t$")
plt.plot(alphas, label=r"$\alpha_t$")
plt.plot(alphas_cumprod, label=r"$\bar{\alpha}_t$")
plt.plot(sqrt_alphas_cumprod, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot(sqrt_one_minus_alphas_cumprod, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$", color='black')
plt.legend(fontsize="x-large")
```
![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015123.png)  

![图 2](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015124.png)  


这个可以配合 guidance `x = (x.detach() + cond_grad * scheduler.alphas_cumprod.sqrt()`, `scheduler.alphas_cumprod.sqrt()`就是系数:
- 对于 shape 的指导，你可能希望大部分效果集中在早期步骤
- 对于 texture 的指导，你可能更希望它们只在生成过程结束时发挥作用。


## 4. UNet

UNet通过多种 embedding 接收更多信息: 
- Time embedding: related to the **timestep and noise level**.
- Context embedding: related to controlling the generation, e.g. text embedding or factor.

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015125.png)

为什么time embedding是加而context embedding是乘？

时间扰动是要作用到unet的layer上的，告诉unet在T时刻处理T时刻的噪声，把Unet从一个纯去噪网络变为了T的条件去噪。
这里对unet参数的改变是经过卷积和池化等操作生成一个和layer一样的张量然后 加 到原来的layer上。
而内容的扰动是增加了不同内容的概率or权重。你可以按照“内容的概率去指导结果中内容生成的概率”去理解，那这里就是需要相乘的。

![训练](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015126.png)

![推理](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015127.png)

- Context is a vector for controlling generation.
- Context can be text embeddings, e.g. > 1000 in length.
- Context can also be categories, one-shot vector, e.g. 5 in length, [0,0,0,1,0]

## 5. DDPM

Each timestep is dependent on the previous one (Markovian)，

The DDPM paper describes a corruption process that adds a small amount of noise for every 'timestep'. 

Given $x_{t-1}$ for some timestep, we can get the next version $x_t$ with:
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015128.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015129.png)

$$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$$


- we take $x_{t-1}$, scale it by $\sqrt{1 - \beta_t}$ and add noise scaled by $\beta_t$. 

$$q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$$

- Markovian的性质, 可以迭代, $q(\mathbf{x}_{2} \vert \mathbf{x}_0) = q(\mathbf{x}_2 \vert \mathbf{x}_{1})q(\mathbf{x}_1 \vert \mathbf{x}_{0})$



$$\begin{aligned}
q(\mathbf{x}_t \vert \mathbf{x}_0) 
&= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, \sqrt{(1 - \bar{\alpha}_t)} \mathbf{I}), \\
\text{where } \bar{\alpha}_t &= \prod_{i=1}^T \alpha_i 
\text{ and } 
\alpha_i = 1-\beta_i
\end{aligned}
$$
- we don't necessariy want to do this operation 500 times to get $x_{500}$ so we have another formula to get $x_t$ for any t given $x_0$: <br><br>

$$\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_{t-1};\tilde{\mu}_t(\mathbf{x}_t,\mathbf{x}_0), \tilde{\beta}_t \mathbf{I}), \\
\text{where } \tilde{\mu}_t(\mathbf{x}_t,\mathbf{x}_0)&=\frac{\sqrt{\bar{\alpha}_t}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
\text{ and }  \\
\tilde{\beta}_{t}&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t}
\end{aligned}
$$


> DDPMs are latent generative models trained to recreate **a fixed forward** Markov chain x1,..., xT.
>
> -- 《Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation》

$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$, 唯一的参数 $\beta_t$ 也是由 scheduler 决定好的定值，所以整个过程是固定的。


## 6. DDIM

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015130.png)

DDIM，采用了别的采样器。

马尔可夫链仅用于概率过程，而DDIM消除了随机性，

DDIM更快，因为可以跳过时间步。

生成质量：500步以下，DDIM更好；500步以上，DDPM更好。


Compared to DDPMs that assume a Markovian noise-injecting forward diffusion process, Denoising diffusion implicit models (DDIMs) [33] assume a **non-Markovian** forward process that has the same marginal distribution as DDPMs, and use its corresponding reverse denoising process for sampling, which enables acceleration of the rather onerous sampling process of DDPMs. 

DDIMs also utilize a deterministic forward-backward process and therefore show nearly-perfect reconstruction ability, which is not the case for DDPMs.


![DAE-Talker: High Fidelity Speech-Driven Talking Face Generation with Diffusion Autoencoder](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015131.png)








## 7. LDM

Latent Diffusion Models a.k.a Stable Diffusion

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015132.png)


![Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015133.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015134.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015135.png)

![Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015136.png)

![NeuralField-LDM:Scene Generation with Hierarchical Latent Diffusion Models](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015137.png)


![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062015138.png)
