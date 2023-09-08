
A simple lookup table (即根据 indice 查其`weight`) that stores embeddings of a fixed dictionary and size (即`num_embeddings, embedding_dim`). 


indices $\to$ word embedding(float vector)
- This module is often used to store word embeddings and retrieve them using indices. 
- The input to the module is a list of indices, and the output is the corresponding word embeddings.

PS: 与`nn.Linear`等价的情况：
    
如果 `nn.Linear(num_embeddings, embedding_dim)` 的输入 `x` 是 one-hot 向量，且 `nn.Linear(x)` 是没有 bias 的, 那么 $f(x)=x * w^{T}$ 就相当于 取对应的 `weight`(shape is `(embedding_dim, num_embeddings)`) 的转置.

## Embedding
```python
torch.nn.Embedding(
    num_embeddings,
    embedding_dim,
    padding_idx=None,
    ...,
    device=None,
    dtype=None,
)
```


Parameters:

-   **num\_embeddings** (int)
    
    size of the dictionary of embeddings

    比如例子中 indices 有10种
    
-   **embedding\_dim** (int)

    the size of each embedding vector

    比如例子中 indices 要被嵌入到 3 维向量
    
-   **padding\_idx** (int, optional)

    If specified, the entries at `padding_idx` do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated during training, i.e. it remains as a fixed “pad”. 
    
    For a newly constructed Embedding, the embedding vector at `padding_idx` will default to all zeros, but can be updated to another value to be used as the padding vector.

Variables:

- **weight** (Tensor)
  
    Float类型的 Tensor
  
    shape (`num_embeddings, embedding_dim`)

    一开始embedding是随机的，在训练的时候会自动更新。the learnable weights of the module initialized from $\mathcal{N}(0, 1)$

    `embedding(input)`的结果即是 `weight` 中对应的 indice .


Forward:
- 需要输入的是整型Tensor
- ndim最低是1，`embedding(torch.LongTensor([1]))`

Examples:

```python
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)     # 字典中有10个词，词向量维度为 3
>>> embedding.weight.shape
torch.Size([10, 3])
>>> embedding.weight
Parameter containing:
tensor([[-0.3832,  0.0220, -1.9188],
        [-0.0632, -1.4126,  0.5855],
        [ 0.8871,  0.1590,  2.3521],
        [-0.1524, -0.6415, -0.2019],
        [-0.9758, -2.0687, -0.8225],
        [ 0.2504, -0.1533, -0.5709],
        [ 0.8386, -0.0411,  0.9411],
        [ 0.2548,  2.0526,  0.7185],
        [-0.1560, -0.2255,  0.1900],
        [ 0.1400,  1.5981, -2.5469]], requires_grad=True)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])      # [2,4]
>>> embedding(input)
tensor([[[-0.0632, -1.4126,  0.5855],                           # [2,4,3]
         [ 0.8871,  0.1590,  2.3521],
         [-0.9758, -2.0687, -0.8225],
         [ 0.2504, -0.1533, -0.5709]],

        [[-0.9758, -2.0687, -0.8225],
         [-0.1524, -0.6415, -0.2019],
         [ 0.8871,  0.1590,  2.3521],
         [ 0.1400,  1.5981, -2.5469]]])

>>> # example with padding_idx
>>> embedding = nn.Embedding(4, 3, padding_idx=0)
>>> embedding.weight
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],        # padding
        [-0.8863,  0.8985,  0.8279],
        [ 0.1615, -1.2078,  0.4549],
        [-0.6428,  0.2937, -0.0068]], requires_grad=True)
>>> input = torch.LongTensor([[0, 2, 3, 1]])
>>> embedding(input)
tensor([[[ 0.0000,  0.0000,  0.0000],       # padding
         [ 0.1615, -1.2078,  0.4549],
         [-0.6428,  0.2937, -0.0068],
         [-0.8863,  0.8985,  0.8279]]], grad_fn=<EmbeddingBackward0>)
>>> 

>>> # example of changing `pad` vector
>>> padding_idx = 1
>>> embedding = nn.Embedding(4, 3, padding_idx=padding_idx)
>>> embedding.weight
Parameter containing:
tensor([[-1.0034,  1.6200,  0.5047],
        [ 0.0000,  0.0000,  0.0000],        # padding
        [-0.3249,  0.0934,  0.1546],
        [ 1.9529, -1.6062, -0.3205]], requires_grad=True)
>>> with torch.no_grad():
...     embedding.weight[padding_idx] = torch.tensor([3, 4, 5]) 
>>> embedding.weight      
Parameter containing:
tensor([[-1.0034,  1.6200,  0.5047],
        [ 3.0000,  4.0000,  5.0000],        # [3,4,5]
        [-0.3249,  0.0934,  0.1546],
        [ 1.9529, -1.6062, -0.3205]], requires_grad=True)
```
## CLASSMETHOD `from_pretrained`
```python
from_pretrained(
    embeddings,
    freeze=True,
    padding_idx=None,
    ...
)
```

Creates Embedding instance from given 2-dimensional FloatTensor.

Parameters:

-   **embeddings** (Tensor)
  
    FloatTensor containing weights for the Embedding. First dimension is being passed to Embedding as `num_embeddings`, second as `embedding_dim`.
    
-   **freeze** (bool, optional)
  
    If `True`, the tensor does not get updated in the learning process. Equivalent to `embedding.weight.requires_grad = False`. Default: `True`

Examples:

```python
>>> # FloatTensor containing pretrained weights
>>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])    # [2, 3]
>>> embedding = nn.Embedding.from_pretrained(weight)
>>> embedding.weight
Parameter containing:
tensor([[1.0000, 2.3000, 3.0000],
        [4.0000, 5.1000, 6.3000]])
>>> input = torch.LongTensor([1])   # [1]
>>> embedding(input)                # [1, 3]                                   
tensor([[4.0000, 5.1000, 6.3000]])
```