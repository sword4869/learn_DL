
[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)


The original StyleGAN2 takes a style vector $w$ as input and uses it to modulate convolution operations in a total of $L$ generative layers during training. During inference, different ws can be used at different layers, formulating the $W^+$ space $W = {w_1, . . . , w_L}$.