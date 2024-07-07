- [1. 增加维度与删除维度](#1-增加维度与删除维度)
  - [1.1. 增加维度 np.newaxis, None, np.expand\_dims](#11-增加维度-npnewaxis-none-npexpand_dims)
  - [1.2. 广播](#12-广播)
    - [1.2.1. 计算shape](#121-计算shape)
    - [1.2.2. 修正shape](#122-修正shape)
      - [1.2.2.1. torch.Tensor.expand](#1221-torchtensorexpand)
  - [1.3. np.repeat](#13-nprepeat)
    - [1.3.1. torch.Tensor.repeat](#131-torchtensorrepeat)
  - [1.4. np.squeeze](#14-npsqueeze)
  - [1.5. 应用维度](#15-应用维度)
  - [1.6. einops](#16-einops)
    - [1.6.1. einops](#161-einops)
    - [1.6.2. torch.einsum](#162-torcheinsum)
- [2. np.concatenate/stack](#2-npconcatenatestack)
  - [2.1. concatenate](#21-concatenate)
  - [2.2. stack](#22-stack)
- [3. np.tile](#3-nptile)
- [4. linspace](#4-linspace)
  - [4.1. torch.linspace](#41-torchlinspace)
- [5. meshgrid](#5-meshgrid)
  - [5.1. torch.meshgrid](#51-torchmeshgrid)
- [6. np.take / ndarray.take](#6-nptake--ndarraytake)




## 1. 增加维度与删除维度

### 1.1. 增加维度 np.newaxis, None, np.expand_dims

- `np.newaxis, None`
  
    是切片, <https://zhuanlan.zhihu.com/p/356601576>
    
```python
>>> np.newaxis is None             # `np.newaxis` 等价于 `None`
True

>>> x = np.array([1, 2])
>>> x[np.newaxis, :].shape
(1, 2)
>>> x[np.newaxis].shape     # 可以简写
(1, 2)
>>> x[:, None].shape               
(2, 1)

>>> y = np.array([[1, 2],[3, 4]])
>>> y[:, :, None].shape            # 多个连续的`:` 等于 `...`
(2, 2, 1)
>>> y[..., None].shape
(2, 2, 1)
```
- `np.expand_dims(a, axis)`
  
    要插入`None`到哪些轴

```python
>>> x = np.array([1, 2])                  # [2]
>>> np.expand_dims(x, axis=1).shape       # 插入一个轴，那么有两个轴, 01
(2, 1)
>>> np.expand_dims(x, axis=0).shape
(1, 2)
>>> np.expand_dims(x, axis=(0, 1)).shape  # 插入两个轴，那么有三个轴, 012
(1, 1, 2)
>>> np.expand_dims(x, axis=(1, 0)).shape
(1, 1, 2)
>>> np.expand_dims(x, axis=(2, 0)).shape
(1, 2, 1)
>>> np.expand_dims(x, axis=(0, 2)).shape
(1, 2, 1)
```

### 1.2. 广播
- 得到的数组将具有与具有最大维度数的输入数组相同的维度数 ndim = max ndim
- 其中每个维度的大小是输入数组中对应维度的
- 它从尾部（即最右侧）尺寸开始，然后向左移动。
  
  即 ndim 较小的数组会在前面追加一个长度为 1 的维度。
- 注意：缺失的尺寸假定为1，即被扩散的轴必须是1

#### 1.2.1. 计算shape 

`numpy.broadcast_shapes(*args)`
```python
>>> np.broadcast_shapes((3,1),(3,))
(3, 3)
# 3 x 1
#     3
# 3 x 3
>>> np.broadcast_shapes((1,3),(3,))
(1, 3)
# 1 x 3
#     3
# 1 x 3
```
`numpy.broadcast` 类，传入参数是 init 函数
```python
>>> x = np.array([[1], [2], [3]])
>>> y = np.array([4, 5, 6])
>>> np.broadcast(x, y)
<numpy.broadcast object at 0x000001F6B7D09D00>
>>>
>>> np.broadcast(x, y).shape
(3, 3)
>>> np.broadcast(x, y).ndim
2
```

!!! note 缺失的尺寸假定为1，即被扩散的轴必须是1
       ```python
       >>> np.broadcast_shapes((4,),(2,))
       ValueError: shape mismatch: objects cannot be broadcast to a single shape.  
       Mismatch is between arg 0 with shape (4,) and arg 1 with shape (2,).

       >>> np.broadcast_shapes((4,3),(3))
       (4, 3)
       >>> np.broadcast_shapes((4,3),(1,3))
       (4, 3)
       >>> np.broadcast_shapes((4,3),(2,3))
       ValueError: shape mismatch: objects cannot be broadcast to a single shape.  
       Mismatch is between arg 0 with shape (4, 3) and arg 1 with shape (2, 3).
       ```
#### 1.2.2. 修正shape
`numpy.broadcast_to(array, shape, subok=False)`
```python
>>> x = np.array([1, 2, 3])
>>> np.broadcast_to(x, (4, 3))
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
```
`numpy.broadcast_arrays(*args, subok=False)`
```python
>>> x = np.array([[1,2,3]])        # (1, 3)
>>> y = np.array([[4],[5]])        # (2, 1)
>>> np.broadcast_arrays(x, y)
[array([[1, 2, 3],
       [1, 2, 3]]), array([[4, 4, 4],
       [5, 5, 5]])]
>>> np.broadcast_arrays(x, y)[0].shape
(2, 3)
>>> np.broadcast_arrays(x, y)[1].shape
(2, 3)
```

!!! note 不匹配尾部

       ```python
       # 我们想要 (3, 28, 28)
       >>> y = np.array([4,5,6])   # [3]
       >>> np.broadcast_to(y, (3, 28, 28))                  
       ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (3,)  and requested shape (3,28,28)
    
       >>> np.broadcast_to(y.reshape(3, 1, 1), (3, 28, 28))
       >>> np.broadcast_to(y[..., None, None], (3, 28, 28))
       ```


##### 1.2.2.1. torch.Tensor.expand
`torch.Tensor.expand(*sizes)`
```python
>>> x = torch.Tensor([[1], [2], [3]])            # [3, 1]
>>> x.expand(3,4)                                # [3, 4]
tensor([[1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.]])
```
### 1.3. np.repeat
`numpy.repeat(a, repeats, axis=None)` or `Ndarry.repeat(repeats, axis=None)`:
- `axis`: 默认`None`展平数组
```python
>>> np.repeat(3, 4)                # 3 重复 4次
array([3, 3, 3, 3])
>>> x = np.array([[1,2],[3,4]])
>>> np.repeat(x, 2)                # [1,2,3,4] 每个重复 2次
array([1, 1, 2, 2, 3, 3, 4, 4])
>>> np.repeat(x, 3, axis=1)        # (2,2) 重复 dim=1 3次，那么(2,6)
array([[1, 1, 1, 2, 2, 2],
       [3, 3, 3, 4, 4, 4]])
>>> np.repeat(x, [1, 2], axis=0)   # (2,2) dim=0的 x[0]重复1次, x[1]重复2次
array([[1, 2],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, [1, 2], axis=1)   # x[:, 0] 重复1次，x[:, 1] 重复2次
array([[1, 2, 2],
       [3, 4, 4]])
```

例子：
```python
>>> a = np.array([1,2,3,4])
>>> np.expand_dims(a, 0).repeat(1000, axis=0).shape # [4]->[1,4]->[1000,4]
(1000, 4)
```

#### 1.3.1. torch.Tensor.repeat
`torch.Tensor.repeat(*sizes)`: 

要求`*sizes` 对齐 tensor变量的n_dim.

- 先补全维度 [1, 1, ..., 原本的维度], 
- 再对每个维度分别复制(a,b,...)次。

而且不是像`numpy.repeat()`每个元素分别复制，其是整块复制（可以理解为**先从最后一维度复制**，复制好后再重复前一维度）

```python
>>> torch.tensor([3]).repeat(4)    # [1]
tensor([3, 3, 3, 3])               # [4] 

>>> a = torch.tensor([1,2])        # [2]
>>> a.repeat(2, 2)                 # [2] -> [1,2] -> [2, 4]
tensor([[1, 2, 1, 2],              # [[1,2]] 从最后一维度复制是 [[1,2,1,2]]，再复制前一维度，[[1,2,1,2],[1,2,1,2]]
        [1, 2, 1, 2]])
# 而不是
# tensor([[1, 1, 2, 2],
#         [1, 1, 2, 2]])
>>> a = torch.tensor([[1,2],[3,4]])
>>> a.repeat(2, 3)
tensor([[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]])
# 先是最后一维度复制3次，[[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]
# 再是前一维度复制2次
```

```python
def noise_like(shape, device):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    return repeat_noise()
```

### 1.4. np.squeeze

删除维度 

`numpy.squeeze(a, axis=None)`

```python
>>> x = np.array([[[0], [1], [2]]])
>>> x.shape
(1, 3, 1)
>>> np.squeeze(x).shape
(3,)
>>> np.squeeze(x, axis=0).shape
(3, 1)
>>> np.squeeze(x, axis=1).shape
Traceback (most recent call last):
...
ValueError: cannot select an axis to squeeze out which has size not equal to one
>>> np.squeeze(x, axis=2).shape
(1, 3)
>>> x = np.array([[1234]])
>>> x.shape
(1, 1)
>>> np.squeeze(x)
array(1234)  # 0d array
>>> np.squeeze(x).shape
()
>>> np.squeeze(x)[()]
1234
```

### 1.5. 应用维度

```python
>>> arr = np.arange(4).reshape((2,2))
>>> arr
array([[0, 1],
       [2, 3]])
>>> arr.sum()
6
>>> arr.sum(axis=0)
array([2, 4])
>>> arr.sum(axis=1)
array([1, 5])
>>> arr.sum(axis=0, keepdims=True)
array([[2, 4]])
```

```python
>>> arr = np.arange(24).reshape((2,3,4))
>>> arr
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
        
# (1) arr[0] + arr[1] 
# (2) arr[0, :, :] + arr[1, :, :]
# (3) 
# output_shape = arr.shape[:axis] + arr.shape[axis+1:]
# output = np.zeros(output_shape)
# for i in range(arr.shape[axis]): 
#      output += arr[i, :, :]      # 这里i还需要变动位置
>>> arr.sum(axis=0)
array([[12, 14, 16, 18],
       [20, 22, 24, 26],
       [28, 30, 32, 34]])
>>> arr.sum(axis=1)
array([[12, 15, 18, 21],
       [48, 51, 54, 57]])
>>> arr.sum(axis=2)
array([[ 6, 22, 38],
       [54, 70, 86]])
```

### 1.6. einops

#### 1.6.1. einops

numpy.ndarray, tensorflow, pytorch, 或者 list.

```python
from einops import repeat, reduce, rearrange
repeat(timesteps, 'b -> b d', d=dim)
```
```python
# list of 32 images in "h w c" format
>>> images = [torch.randn(30, 40, 3) for _ in range(32)]

# 即 torch.stack(images), output is a single array
>>> rearrange(images, 'b h w c -> b h w c').shape
(32, 30, 40, 3)

# stack 且 reordered axes to "b c h w"
>>> rearrange(images, 'b h w c -> b c h w').shape
(32, 3, 30, 40)

# concatenate images along height (vertical axis), 960 = 32 * 30
>>> rearrange(images, 'b h w c -> (b h) w c').shape
(960, 40, 3)

# concatenated images along horizontal axis, 1280 = 32 * 40
>>> rearrange(images, 'b h w c -> h (b w) c').shape
(30, 1280, 3)

# flattened each image into a vector, 3600 = 30 * 40 * 3
>>> rearrange(images, 'b h w c -> b (c h w)').shape
(32, 3600)

# split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
>>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
(128, 15, 20, 3)

# space-to-depth operation
>>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
(32, 15, 20, 12)
```


#### 1.6.2. torch.einsum

> 矩阵的乘积

两个矩阵, $A \in \mathbb{R}^{I \times K}$, $B \in \mathbb{R}^{K \times J}$ ，两个矩阵的乘积 $C \in \mathbb{R}^{I \times J}$

用爱因斯坦求和约定可以如下表示：

$$ C_{ij} = (AB)_{ij}= \sum_{k=1}^{K}A_{ik} B_{kj} $$

```python
>>> a = torch.arange(4).reshape(2,2)
>>> b = torch.arange(4,8).reshape(2,2)
>>> a
tensor([[0, 1],
        [2, 3]])
>>> b
tensor([[4, 5],
        [6, 7]])
>>> torch.einsum('ij,jk->ik', a, b)
tensor([[ 6,  7],
        [26, 31]])
>>> a @ b
tensor([[ 6,  7],
        [26, 31]])
```

$A B^\top$

```python
>>> torch.einsum('ik,jk->ij', a, b)
tensor([[ 5,  7],
        [23, 33]])
>>> a @ b.t()
tensor([[ 5,  7],
        [23, 33]])
```

$A^\top B$
```python
>>> torch.einsum('ki,kj->ij', a, b)
tensor([[12, 14],
        [22, 26]])
>>> a.t() @ b
tensor([[12, 14],
        [22, 26]])
```
> 求和

```python
>>> torch.einsum('ij->', a)
tensor(6)

# 行求和
>>> torch.einsum('ij->i', a)
tensor([1, 5])

# 列求和
>>> torch.einsum('ij->j', a)
tensor([2, 4])
```

> 转置

```python
>>> torch.einsum('ij->ji', a)
tensor([[0, 2],
        [1, 3]])
```

## 2. np.concatenate/stack
### 2.1. concatenate
The arrays must have the same shape, except in the dimension corresponding to axis (the first, by default).
```python
# numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")

# a: [2, 2], b: [1, 2]
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# [2+1, 2] = [3, 2]
np.concatenate([a, b])
# array([[1, 2],
#        [3, 4],
#        [5, 6]])

# [2, 2+1] = [2, 3]
np.concatenate((a, b.T), axis=1)
# array([[1, 2, 5],
#        [3, 4, 6]])

# If `axis` is `None`, arrays are flattened before use. 
np.concatenate((a, b), axis=None)
# array([1, 2, 3, 4, 5, 6])
```
### 2.2. stack
stack
```python
>>> arrays = [np.random.randn(3, 4) for _ in range(10)]
>>> np.stack(arrays, axis=0).shape
(10, 3, 4)

>>> np.stack(arrays, axis=1).shape
(3, 10, 4)

>>> np.stack(arrays, axis=2).shape
(3, 4, 10)

>>> a = np.array([1, 2, 3])
>>> b = np.array([4, 5, 6])
>>> np.stack((a, b))
array([[1, 2, 3],
       [4, 5, 6]])

>>> np.stack((a, b), axis=-1)
array([[1, 4],
       [2, 5],
       [3, 6]])
```
vstack: stack along the first axis
```python
#### stack
>>> a = np.array([1, 2, 3])        # [3]
>>> b = np.array([4, 5, 6])        # [3]
>>> np.vstack((a,b))               # [2, 3]
array([[1, 2, 3],
       [4, 5, 6]])

#### concatenate
>>> a = np.array([[1], [2], [3]])  # [3, 1]
>>> b = np.array([[4], [5], [6]])  # [3, 1]
>>> np.vstack((a,b))               # [6, 1]
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
```
hstack: stack along the second axis, except for 1-D arrays where it concatenates along the first axis.
```python
#### append
>>> a = np.array((1,2,3))          # [3]
>>> b = np.array((4,5,6))          # [3]
>>> np.hstack((a,b))               # [6]
array([1, 2, 3, 4, 5, 6])

#### concatenate
>>> a = np.array([[1],[2],[3]])    # [3, 1]
>>> b = np.array([[4],[5],[6]])    # [3, 1]
>>> np.hstack((a,b))               # [3, 2]
array([[1, 4],
       [2, 5],
       [3, 6]])
```
## 3. np.tile

`numpy.tile(A, reps)`:通过重复A代表给出的次数来构建数组，平铺效果。
- If `reps` has length `d`, the result will have dimension of `max(d, A.ndim)`.
- 具体是，对`A`和`reps`的shape都来对齐，在前面补1，实现`d`相等，`[2, 2], [3]`→`[2, 2], [1, 3]`。
- `reps`对齐后的意思就是，对相应的维度进行复制几次。
```python
#########
# [3]
a = np.array([0, 1, 2])
# 都是1维，那么对1维复制2次
np.tile(a, 2)
array([0, 1, 2, 0, 1, 2])

# [3]变成[1, 3]，即[[0,1,2]]
# 然后第一维复制2次得到[[0,1,2],[0,1,2]]，
# 然后再对第二维度复制2次
np.tile(a, (2, 2))
array([[0, 1, 2, 0, 1, 2],
       [0, 1, 2, 0, 1, 2]])

# [3]变成[1,1,3]，即[[[0,1,2]]]
# 然后第一维复制2次得到[[[0,1,2]],[[0,1,2]]]]
# 然后再对第二维度复制1次则不变，
# 第三维度复制2次即下
np.tile(a, (2, 1, 2))
array([[[0, 1, 2, 0, 1, 2]],
       [[0, 1, 2, 0, 1, 2]]])

#########
# [2,2]
b = np.array([[1, 2], [3, 4]])

# [2]变成[1,2]
# 第一维复制1次则不变，[[1, 2], [3, 4]]
# 第二维复制2次，则
np.tile(b, 2)
array([[1, 2, 1, 2],
       [3, 4, 3, 4]])

# 维度相等
# 第一维复制2次则，[[1, 2], [3, 4], [1, 2], [3, 4]]
# 第二维复制1次，则不变
np.tile(b, (2, 1))
array([[1, 2],
       [3, 4],
       [1, 2],
       [3, 4]])

#########
[4]
c = np.array([1,2,3,4])
# [4]变成[1,4]，即[[1,2,3,4]]
# 第一维复制4次则，[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# 第二维复制1次，则不变
np.tile(c,(4,1))
array([[1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4]])
```

## 4. linspace

在区间内，平均划分，返回n个点。
```python
# 包含 start 和 stop, [start, stop]
>>>  np.linspace(1, 10, 10)
array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

# 不想在序列计算中包括最后一点, [start, stop)
>>> np.linspace(1, 10, 10, endpoint=False)
array([1. , 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1])
```


- 前者：
       
    $(\text{start}, \text{start} + \dfrac{\text{end} - \text{start}}{\text{steps} - 1}, \ldots, \text{start} + (\text{steps} - 2) * \dfrac{\text{end} - \text{start}}{\text{steps} - 1}, \text{end})$
- 后者：
  
    $(\text{start}, \text{start} + \dfrac{\text{end} - \text{start}}{\text{steps}}, \ldots, \text{start} + (\text{steps} - 2) * \dfrac{\text{end} - \text{start}}{\text{steps}}, \text{start} + (\text{steps} - 1) * \dfrac{\text{end} - \text{start}}{\text{steps}})$

例子：
- `array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])`

`np.linspace(1, 10, 10)`

- `array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])`

`np.linspace(0, 10, 10 + 1)`
- `array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])`

`np.linspace(0, 9, 10)` or `np.linspace(0, 10, 10 + 1)[:-1]` or `np.linspace(0, 10, 10, endpoint=False)`

### 4.1. torch.linspace

`torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`

只有包含 start 和 stop, [start, stop] 的模式了。

## 5. meshgrid


参考资料：[meshgrid理解](https://blog.csdn.net/lllxxq141592654/article/details/81532855), [numpy](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html)


`numpy.meshgrid(*xi, copy=True, sparse=False, indexing='xy')`


1. meshgrid函数的作用：生成坐标矩阵。
2. meshgrid函数的输入，K个一维数组
3. meshgrid函数的输出：K个K维矩阵, 分别表示第一维度、第二维度……第K维度。


> `indexing='xy'`: 
> - Cartesian indexing. 
> - M个横坐标，N个纵坐标。返回成数组自然是M列N行
> - 逐行

- In the 2-D case ：inputs length (M, N), outputs shape (N, M) 
- In the N-D case : inputs length (M, N, P3, P4, ..., PK), outputs shape (N, M, P3, P4, ..., PK)

```python
import numpy as np

# 横坐标3个，纵坐标2个
nx, ny = (3, 2)
x = np.linspace(0, 1, nx)   # 3个
y = np.linspace(0, 1, ny)   # 2个

# input length (3, 2)， 横坐标3个，纵坐标2个
# output shape (2, 3)， 2行3列
xv, yv = np.meshgrid(x, y)

# 逐行，自然横坐标xv, 行内递增，每行都相同；纵坐标yv，每行都是同一个纵坐标，每行递增
>>> xv
array([[0. , 0.5, 1. ],
        [0. , 0.5, 1. ]])
>>> yv 
array([[0.,  0.,  0.],
       [1.,  1.,  1.]])

```
![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062019760.png)  

```python
############# indexing='xy': x个横坐标，y个纵坐标；x列y行
>>> nx, ny = 3, 2
>>> np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='xy')
[array([[0. , 0.5, 1. ],
       [0. , 0.5, 1. ]]), array([[0., 0., 0.],
       [1., 1., 1.]])]
>>> coords = np.stack(np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='xy'), -1)
# 将x坐标和y坐标拼接起来: [2, 3, 2], [y个纵坐标, x个横坐标, xy坐标]
>>> coords
array([[[0. , 0. ],
        [0.5, 0. ],
        [1. , 0. ]],

       [[0. , 1. ],
        [0.5, 1. ],
        [1. , 1. ]]])
# 将坐标串联起来: 逐行，x从小到大，y再从小到大
>>> coords.reshape([-1, coords.shape[-1]])
array([[0. , 0. ],
       [0.5, 0. ],
       [1. , 0. ],
       [0. , 1. ],
       [0.5, 1. ],
       [1. , 1. ]])
```

> `indexing='ij'`: 
> - matrix indexing. 
> - M行N列的矩阵
> - 逐列

- In the N-D case : inputs length (M, N, P3, P4, ..., PK), outputs shape (M, N, P3, P4, ..., PK)

```python
import numpy as np

x = np.linspace(0, 1, 4)
y = np.linspace(0, 1, 3)
z = np.linspace(0, 1, 2)

xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
print(xv.shape)
print(yv.shape)
print(zv.shape)
# (4, 3, 2)
# (4, 3, 2)
# (4, 3, 2)


############## xv只在dim=0即x轴变，yv只在dim=1即y轴变, ...; 其他轴都是同一个数
>>> xv
array([[[0.        , 0.        ],
        [0.        , 0.        ],
        [0.        , 0.        ]],

       [[0.33333333, 0.33333333],
        [0.33333333, 0.33333333],
        [0.33333333, 0.33333333]],

       [[0.66666667, 0.66666667],
        [0.66666667, 0.66666667],
        [0.66666667, 0.66666667]],

       [[1.        , 1.        ],
        [1.        , 1.        ],
        [1.        , 1.        ]]])
>>> yv
array([[[0. , 0. ],
        [0.5, 0.5],
        [1. , 1. ]],

       [[0. , 0. ],
        [0.5, 0.5],
        [1. , 1. ]],

       [[0. , 0. ],
        [0.5, 0.5],
        [1. , 1. ]],

       [[0. , 0. ],
        [0.5, 0.5],
        [1. , 1. ]]])
>>> zv
array([[[0., 1.],
        [0., 1.],
        [0., 1.]],

       [[0., 1.],
        [0., 1.],
        [0., 1.]],

       [[0., 1.],
        [0., 1.],
        [0., 1.]],

       [[0., 1.],
        [0., 1.],
        [0., 1.]]])
>>> xv[0, :, :], xv[1, :, :]
(array([[0., 0.],
       [0., 0.],
       [0., 0.]]), array([[0.33333333, 0.33333333],
       [0.33333333, 0.33333333],
       [0.33333333, 0.33333333]]))
>>> yv[:, 0, :], yv[:, 1, :]
(array([[0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.]]), array([[0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5]]))
>>> zv[:,:,0], zv[:,:,1]
(array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]]), array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]]))

######## 所以，等同于对一维数组扩展维度
>>> from einops import repeat
>>> xv2 = repeat(np.linspace(0, 1, 4), 'x -> x 3 2')
>>> yv2 = repeat(np.linspace(0, 1, 3), 'y -> 4 y 2')
>>> zv2 = repeat(np.linspace(0, 1, 2), 'z -> 4 3 z')
```

- In the 2-D case ：inputs length (M, N), outputs shape (M, N) 
```python
import numpy as np

# 矩阵是3行2列
nx, ny = (3, 2)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

# input length (3, 2)， 矩阵是3行2列
# output shape (3, 2)， 矩阵是3行2列
xv, yv = np.meshgrid(x, y, indexing='ij')

# 逐列：自然横坐标xv每行都是同一个数，纵坐标yv每行都相同。
# 逐列：自然横坐标xv, 每行都是同一个横坐标，每行递增；纵坐标yv，行内递增，每行都相同
>>> xv
[[0.  0. ]
 [0.5 0.5]
 [1.  1. ]]
>>> yv
[[0. 1.]
 [0. 1.]
 [0. 1.]]
```
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062019761.png)
```python
############# indexing='ij': i行j列
>>> nx, ny = 3, 2
>>> np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
[array([[0. , 0. ],
       [0.5, 0.5],
       [1. , 1. ]]), array([[0., 1.],
       [0., 1.],
       [0., 1.]])]
>>> coords = np.stack(np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij'), -1)
# [3, 2, 2], [j个横坐标, i个纵坐标, xy坐标]
>>> coords
array([[[0. , 0. ],
        [0. , 1. ]],

       [[0.5, 0. ],
        [0.5, 1. ]],

       [[1. , 0. ],
        [1. , 1. ]]])
# 将坐标串联起来: 逐列，y从小到大，x再从小到大
>>> coords.reshape([-1, coords.shape[-1]])
array([[0. , 0. ],
       [0. , 1. ],
       [0.5, 0. ],
       [0.5, 1. ],
       [1. , 0. ],
       [1. , 1. ]])
```



> 例子

例子1： for
```python
xv, yv = np.meshgrid(x, y, indexing='xy')
for i in range(nx):
    for j in range(ny):
        # treat xv[j,i], yv[j,i]

xv, yv = np.meshgrid(x, y, indexing='ij')
for i in range(nx):
    for j in range(ny):
        # treat xv[i,j], yv[i,j]
```

例子2：计算$x^2 + y^2$
```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-5, 5, num=10)
y = np.linspace(-5, 5, num=10)
xv, yv = np.meshgrid(x, y, indexing='xy')
z = xv ** 2 + yv ** 2
print(z)
'''
[[50. 41. 34. 29. 26. 25. 26. 29. 34. 41. 50.]
 [41. 32. 25. 20. 17. 16. 17. 20. 25. 32. 41.]
 [34. 25. 18. 13. 10.  9. 10. 13. 18. 25. 34.]
 [29. 20. 13.  8.  5.  4.  5.  8. 13. 20. 29.]
 [26. 17. 10.  5.  2.  1.  2.  5. 10. 17. 26.]
 [25. 16.  9.  4.  1.  0.  1.  4.  9. 16. 25.]
 [26. 17. 10.  5.  2.  1.  2.  5. 10. 17. 26.]
 [29. 20. 13.  8.  5.  4.  5.  8. 13. 20. 29.]
 [34. 25. 18. 13. 10.  9. 10. 13. 18. 25. 34.]
 [41. 32. 25. 20. 17. 16. 17. 20. 25. 32. 41.]
 [50. 41. 34. 29. 26. 25. 26. 29. 34. 41. 50.]]
'''
ax = plt.axes(projection='3d')
ax.plot_surface(xv, yv, z)
plt.show()
```
![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062019762.png)  


### 5.1. torch.meshgrid

`torch.meshgrid(*tensors, indexing='ij')`

一样，只不过torch的默认值是 `ij` matrix indexing.

<https://blog.csdn.net/weixin_39504171/article/details/106356977>
<https://pytorch.org/docs/stable/generated/torch.meshgrid.html>

## 6. np.take / ndarray.take

`numpy.take(a, indices, axis=None, out=None, mode='raise')[source]
`
- `axis`
  
   The axis over which to select values. By default, **the flattened input** array is used.

`np.take(arr, indices, axis=3)` is equivalent to `arr[:,:,:,indices,...]`.

```python
>>> a = [4, 3, 5, 7, 6, 8]
>>> np.take(a, [0, 1, 4])
array([4, 3, 6])
>>> np.array(a)[[0, 1, 4]]         # 等价
array([4, 3, 6])
>>> np.take(a, [[0, 1], [2, 3]])
array([[4, 3],
       [5, 7]])
>>> np.array(a)[[[0, 1], [2, 3]]]  # 等价
array([[4, 3],
       [5, 7]])
```
```python
>>> b = [[ 3,  4,  5], [ 6,  7,  8], [ 9, 10, 11]]
>>> np.take(b, [2, 1, 0, 4])     # flatten
array([[5, 4, 3, 7]])

>>> np.take(b, [2, 1, 0], axis=0)  # 整行
array([[ 9, 10, 11],
       [ 6,  7,  8],
       [ 3,  4,  5]])
>> np.array(b)[[2,1,0], :]         # 等价
array([[ 9, 10, 11],
       [ 6,  7,  8],
       [ 3,  4,  5]])

>>> np.take(b, [2, 1, 0], axis=1)  # 整列
array([[ 5,  4,  3],
       [ 8,  7,  6],
       [11, 10,  9]])
>> np.array(b)[:, [2,1,0]]         # 等价
array([[ 5,  4,  3],
       [ 8,  7,  6],
       [11, 10,  9]])
```