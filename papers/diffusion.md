
Diffusion Probabilistic Model (DPM) 简称 Diffusion Model

Denoising Diffusion Probabilistic Models (DDPM) ，多了个 Denoising.


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