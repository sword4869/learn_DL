- [条件 diffusion](#条件-diffusion)
- [1. Overview](#1-overview)
- [2. Component](#2-component)
  - [2.1. VAE](#21-vae)
    - [2.1.1. latent space](#211-latent-space)
    - [2.1.2. details](#212-details)
  - [2.2. CLIP](#22-clip)
  - [UNet](#unet)
- [3. 训练和推理](#3-训练和推理)
  - [3.1. 训练](#31-训练)
  - [3.2. 推理](#32-推理)
- [4. Text conditioning](#4-text-conditioning)


---

## 条件 diffusion


`model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample`
- `noisy_latents, timesteps` 是标准的unet输入，
- 而 `encoder_hidden_states` 是额外的条件。

## 1. Overview

Stable diffusion 并不是单个模型，而是由三个模型组合起来的。


- Text Encoder (pre-tained 好的Clip中的Encoder)

    作为unet的条件

    Input: text.

    Output: **token embeddings vectors**, **77 tokens** each in 768 dimensions.

- Image Information Creator (ldm中的 UNet + Scheduler)

    gradually denoising process information in the information (latent) space.
    
    Input: token embeddings and a random noise tensor.

    Output: A processed information tensor (4,64,64)

- VAE

latent 就是中间产物。

## 2. Component
### 2.1. VAE

#### 2.1.1. latent space

压缩，尺寸更小。image size $(H,W,3)$, latent size $(h,w,c)$, 因子$f = \dfrac{H}{h}=\dfrac{W}{w}$, $f$ 是 8, 所以训练的图片的尺寸要可以整除 8 .


the diffusion process in the 'latent space' of the VAE. 不是直接在 pixel space 上扩散图片，而是在 latent space 上扩散 latent。

- efficient，参数量相对于直接 pixel space 更少，使得stable diffusion(latent diffusion)快
- 不仅仅局限于图片了, 可以将其他类型的数据转化为 latent.

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017878.png)

Encoder-Decoder架构，Encoder得到的中间产物，经过Diffusion处理后，传给Decoder。

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017879.png)

原本 pixel space 上 噪声和图片一样大小 $(H,W)$，现在 latent space 上 噪声和 latent 一样大小 $(h,w)$

![High-Resolution Image Synthesis with Latent Diffusion Models. Fig 3](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017880.png)

#### 2.1.2. details

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017881.png)


- Encoder

    compress the image to latent representation
    
    Input:  The training image (3, 512, 512)
    
    Output: latent (4,64,64)

- Decoder

    paints the final image.
    
    Input:  latent (4,64,64)
    
    Output: The resulting image (3, 512, 512).

### 2.2. CLIP
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017882.png)

string $\to$ *`pipe.tokenizer`* $\to$ tokens $\to$ *`pipe.text_encoder`* $\to$ text_embedding



### UNet




## 3. 训练和推理

### 3.1. 训练

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017878.png)

重建原始图片。

### 3.2. 推理
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017883.png)

1. 代替encoder：从高斯分布采样，$(h,w)$ latent 大小的噪声
2. 传给unet
3. 最终，多次迭代后unet输出的中间产物，交给Decoder

## 4. Text conditioning

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062017884.png)

text embedding 在每步都会传给unet。
