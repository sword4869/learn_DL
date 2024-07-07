## intro
CLIP (Contrastive Language–Image Pre-training)

CLIP can be instructed in natural language to perform a great variety of classification benchmarks, without directly optimizing for the benchmark’s performance, similar to the “zero-shot” capabilities of GPT-25 and GPT-3.


![图 3](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013291.png)  



## Training

![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062013292.png)  

训练数据：image-text pair
网络结构：两个 Encoder, image encoder 和 text encoder

1. image 通过 image ecoder 的 image embedding
2. text 通过 text ecoder 的 text embedding
3. 评估 image embedding 和 text embedding 的相似度。越相似， CLIP Score越高。
4. 根据相似程度来更新两个Encoder。

不仅有正相关的相似，还有负样本的不相似。


## Inference

Stable Diffusion 就拿 CLIP 的 pre-trained **Text Encoder** 去用：将文本通过得到 text embedding.

Input: text.
Output: token embeddings vectors, 77 token each in 768 dimensions (77 x 768).

## code
```bash
pip install open_clip_torch
```
```python
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model.to(device)

# Transforms to resize and augment an image + normalize to match CLIP's training data
tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),  # Random CROP each time
        transforms.RandomAffine(
            5
        ),  # One possible random augmentation: skews the image
        transforms.RandomHorizontalFlip(),  # You can add additional augmentations if you like
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

# And define a loss function that takes an image, embeds it and compares with
# the text features of the prompt
def clip_loss(image, text_features):
    image_features = clip_model.encode_image(
        tfms(image)
    )  # Note: applies the above transforms
    input_normed = F.normalize(image_features.unsqueeze(1), dim=2)
    embed_normed = F.normalize(text_features.unsqueeze(0), dim=2)
    dists = (
        input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
    )  # Squared Great Circle Distance
    return dists.mean()
```
```python
prompt = "Red Rose (still life), red flower painting"
text_tokens = open_clip.tokenize([prompt]).to(device)
text_features = clip_model.encode_text(text_tokens)
```