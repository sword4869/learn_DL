


---


Diffusion Probabilistic Model (DPM) 简称 Diffusion Model

Denoising Diffusion Probabilistic Models (DDPM) ，多了个 Denoising.


---

时间步决定了噪声的等级。

## forward process

forward process: 给图片随时间步加噪，[1,2,3,...,T].

## training 

神经网络学习噪声的分布，什么是图片上的噪声、什么不是图片的噪声

![Alt text](image-16.png)

更稳定的训练策略：不是对同一张图片施加随时间步增长（遍历所有的时间步）的噪声等级，而是每次随机取一张图片，分别对其采**随机的时间步**的噪声等级。

## sample process

一个时间步表示一次迭代

![Alt text](image-14.png)

分解1：

- ![Alt text](image-12.png)

- Noise sample 减去 神经网络预测出来的 Predicted Noise, 得到 更清晰的图像。 `sample = sample - predicted_noise`

- 但这样的结果得到的只是乱七八糟的图片。

分解2：
再加入 extra_noise，ddpm scheduler再预测出三个缩放因子，协调 sample、predicted_noise、extra_noise。

![Alt text](image-13.png)


## UNet

The UNet can take in more information in the form of embeddings

- Time embedding: related to the **timestep and noise level**.
- Context embedding: related to controlling the generation, e.g. text description or factor (more later).

![Alt text](image-15.png)

为什么timeembedding是加而contextembedding是乘？
时间扰动是要作用到unet的layer上的，告诉unet在T时刻处理T时刻的噪声，把Unet从一个纯去噪网络变为了T的条件去噪。
这里对unet参数的改变是经过卷积和池化等操作生成一个和layer一样的张量然后 加 到原来的layer上。
而内容的扰动是增加了不同内容的概率or权重。你可以按照“内容的概率去指导结果中内容生成的概率”去理解，那这里就是需要相乘的。

![Alt text](image-17.png)

![Alt text](image-18.png)

- Context is a vector for controlling generation.
- Context can be text embeddings, e.g. > 1000 in length.
- Context can also be categories, one-shot vector, e.g. 5 in length, [0,0,0,1,0]

## DDPM

Each timestep is dependent on the previous one (Markovian)，






The DDPM paper describes a corruption process that adds a small amount of noise for every 'timestep'. 

Given $x_{t-1}$ for some timestep, we can get the next version $x_t$ with:

$$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad$$

$$q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$$

- we take $x_{t-1}$, scale it by $\sqrt{1 - \beta_t}$ and add noise scaled by $\beta_t$. 
- $\beta$ is defined for every timestep $t$ accoridng to some schedule, and determines how much noise is added per timestep. 

Now, we don't necessariy want to do this operation 500 times to get $x_{500}$ so we have another formula to get $x_t$ for any t given $x_0$: <br><br>

$$\begin{aligned}
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, \sqrt{(1 - \bar{\alpha}_t)} \mathbf{I})
\end{aligned}, \text{where } \bar{\alpha}_t = \prod_{i=1}^T \alpha_i \text{ and } \alpha_i = 1-\beta_i$$

- We can plot $\sqrt{\bar{\alpha}_t}$ (labelled as `sqrt_alpha_prod`) and $\sqrt{(1 - \bar{\alpha}_t)}$ (labelled as `sqrt_one_minus_alpha_prod`)

## DDIM

![Alt text](image-19.png)

DDIM，采用了别的采样器。

马尔可夫链仅用于概率过程，而DDIM消除了随机性，

DDIM更快，因为可以跳过时间步。

生成质量：500步以下，DDIM更好；500步以上，DDPM更好。

## stable diffusion

不是直接在图片上操作，而是在 image embedding 上操作。

## other

text conditioning

cross-attention

classifier free guidance

---

Denoising diffusion probabilistic models (DDPMs) [9] associate image generation with the sequential denoising process of isotropic Gaussian noise. The model is trained to predict the noise from the input image. Unlike other generative models such as GANs and most traditional-style VAEs that encode input data in a lowdimensional space, diffusion models have a latent space that is the same size as the input. Although DDPMs require a lot of feed-forward steps to generate samples, their image fidelity and diversity are superior to other types of generative models. Compared to DDPMs that assume a Markovian noise-injecting forward diffusion process, Denoising diffusion implicit models (DDIMs) [33] assume a non-Markovian forward process that has the same marginal distribution as DDPMs, and use its corresponding reverse denoising process for sampling, which enables acceleration of the rather onerous sampling process of DDPMs. DDIMs also utilize a deterministic forward-backward process and therefore show nearly-perfect reconstruction ability, which is not the case for DDPMs.

---

![DAE-Talker: High Fidelity Speech-Driven Talking Face Generation with Diffusion Autoencoder](image.png)


![Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](image-1.png)

![Alt text](image-2.png)





## LDM

Latent Diffusion Models a.k.a Stable Diffusion

![Alt text](image-3.png)


![Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](image-5.png)

![Alt text](image-6.png)

![Alt text](image-7.png)

![Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](image-8.png)

![Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation](image-9.png)

![Alt text](image-10.png)

![NeuralField-LDM:Scene Generation with Hierarchical Latent Diffusion Models](image-11.png)

LDM: which first construct an intermediate latent distribution of the training data then fit a diffusion model on the latent distribution


