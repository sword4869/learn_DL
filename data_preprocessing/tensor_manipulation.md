[toc]
# tensor_manipulation

## create


```python
import torch



# 标量
x = torch.tensor(8)
# 行向量
torch.arange(4)
# tensor([0, 1, 2, 3])
torch.arange(4.0)
# tensor([0., 1., 2., 3.])
torch.arange(0.7, 4.7)
# tensor([0.7000, 1.7000, 2.7000, 3.7000])


### full

torch.full([3, 4], 0)

# 全0
# torch.zeros((2, 3, 4)),元组格式也行
torch.zeros(2, 3, 4)
torch.zeros(X.shape)
torch.zeros_like(X)

# 全1
torch.ones(2, 3, 4)
torch.ones_like(X)

# 单位矩阵
torch.eye(3, 3)


### 随即
# 均匀分布[0, 1]中随机采样
torch.rand(3, 4)
# rand-normal, 从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
torch.randn(3, 4)

# 限定元素大小 [min, max)
torch.randint(10, 12, (3, 4))   
# 默认最小0
torch.randint(2, (3,))    # 不同于 numpy， size 只接受元组类型

# normal
torch.normal(0, 1, (3, 3))


# Python列表
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```python
# 清零
>>> X.zero_()
tensor([[0., 0., 0.],
        [0., 0., 0.]])
>>> X
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

浮点数
```python
# 直接tensor，浮点数默认 torch.float32
a1 = torch.tensor(12.1)

# 转化整数为浮点数 torch.float32
a2 = torch.tensor(12).float()

# 转化 numpy ，浮点数默认 torch.float64
a3 = torch.tensor(np.array([1.2]))


### 所以，一般在转化时，指定float32
# 方式1
b = torch.tensor(np.array([1.2]), dtype=torch.float32)
# 方式2
c = torch.tensor(np.array([1.2])).float()
```

整数
```python
# 整数默认torch.int64
a1 = torch.tensor(12).dtype

# 转化浮点数为整数 torch.int64
a2 = torch.tensor(12.1).long()

# 转化 numpy ，浮点数默认 torch.int32
a3 = torch.tensor(np.array([12]))
```
```python
>>> torch.tensor([1.,2.,3.]).dtype                  # 默认浮点数是 torch.float32
torch.float32
>>> torch.tensor([1,2,3]).dtype                     # 默认整数是 torch.int64
torch.int64
>>> torch.tensor([1,2,3], dtype=torch.int).dtype    # torch.int 是 torch.int32
torch.int32
```

[`torch.tensor()`注意不要写成 torch.Tensor()](https://blog.csdn.net/weixin_42018112/article/details/91383574)
```python
### 参数类型的不同
# 随机数字
>>> torch.Tensor(2)
tensor([-1.4648e+01,  6.2454e-39])
>>> torch.Tensor([2])
tensor([2.])
>>> torch.Tensor(2,3)
tensor([[1.8217e-35, 0.0000e+00, 1.9274e-38],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])


# 整型
>>> torch.LongTensor([1,2,3])
tensor([1, 2, 3])       # torch.int64

# 浮点型
>>> torch.FloatTensor([1,2,3])
tensor([1., 2., 3.])    # torch.float32
>>> torch.Tensor([1,2,3])
tensor([1., 2., 3.])    # torch.float32
>>> type(torch.Tensor([1,2,3])) == type(torch.FloatTensor([1,2,3]))
True

# TypeError: new(): data must be a sequence (got float)
>>> x = 2.1
>>> torch.Tensor(x) # 报错
>>> torch.Tensor([x])
```
torch.Tensor是类型
```python
>>> isinstance(1, torch.Tensor)
False
>>> isinstance(torch.tensor(1), torch.Tensor)
True
>>> isinstance(torch.LongTensor(1), torch.Tensor)
True
>>> isinstance(torch.LongTensor([1]), torch.Tensor)
True
```


维度

```python
# 0-D标量
x = torch.tensor(8).shape
# torch.Size([])
len(x)  # 0

# 1-D向量
y = torch.arange(12).shape
# torch.Size([12])
len(y)  # 1

z = torch.arange(12).reshape(2, 2, 3).shape
# torch.Size([2, 2, 3])
len(z)  # 3

# 所以，对于非标量，常用
len(X) == X.shape[0]
```
```python
X = torch.arange(8).reshape(2, 4)

# 自动计算出维度
X.reshape(-1, 4)
X.reshape(2, -1)

X.reshape(-1)       # 化为1D
# torch.Size([8])

X.reshape(-1, 8)    # 化为2D
# torch.Size([1, 8])

X.reshape(-1, 1)    # 化为2D
# torch.Size([8, 1])
```

切片

```python
X = torch.arange(36).reshape(4, 3, 3)

X, X[-1], X[1:3], X[1:3, :], X[1,1]

# 超出索引
# X[5], X[5, :] 不行，当行必须是slice格式的，而不是index
print(X[5:6], X[:, 7:8])
# tensor([], size=(0, 3, 3), dtype=torch.int64) 
# tensor([], size=(4, 0, 3), dtype=torch.int64)

# 效果一样
X[0, 0, 0] == X[0][0][0]    # 拆分成index
X[1:3] == X[1:3, :]

# slice的效果不同
X[1, :].shape
# torch.Size([3, 3])
X[1:2, :].shape
# torch.Size([1, 3, 3])
```
为多个元素赋值相同的值
```python
X[0:2, :] = 12
X[:] = 12
```
广播机制

```python
# 形状不能是 (3), 得是 (3,1)，len(shape)要一致
X = torch.arange(3).reshape((3, 1))
Y = torch.arange(2).reshape((1, 2))

a = X+Y
# - (3,1),(1,2),都扩展为(3,2). 
# - 矩阵`X`将复制列，变成`[[0,0], [1,1], [2,2]]`.
# - 矩阵`Y`将复制行，变成`[[0,1], [0,1], [0,1]]`
# - 然后相加
```


## calculate
### multiple
element-wise
```python
X = torch.arange(9).reshape((3, 3))

2 + X
2 * X
2 ** X
```
matrix
```python
X = torch.arange(9).reshape((3, 3))
Y = torch.arange(9).reshape((3, 3))

X + Y
X - Y
X / Y
X * Y   # element-wise, 按元素相乘，即Hadamard product
torch.mul(X, Y)   # 也是element-wise
X ** Y  # 求幂运算
torch.exp(X)  # e^x
```


1D向量的点积：结果是一个0维标量
```python
A = torch.tensor([1, 2, 3])
B = torch.tensor([2, 0, 0])

# (1) torch.dot 限定 1D 向量
torch.dot(A, B) 
# tensor(2)


# (2) element-wise，再求和
(A * B).sum()

# (3)
torch.matmul(A, B)

# (4) 
A @ B
```

2D矩阵-1D向量积：结果是一个1维向量

```python
A = torch.arange(9).reshape(3,3)
x = torch.tensor([1,2,3])

# (1) 限定 matrix-vector (vector-matrix 不能颠倒)
torch.mv(A, x)
# tensor([ 8, 26, 44])

# (2) 
torch.matmul(A, x)

# (3)
A@x
```
矩阵-矩阵乘法: 结果是2D
```python
X = torch.arange(9).reshape((3, 3))
Y = torch.ones_like(X)

# (1) 限定 matrix-matrix multiplication
torch.mm(X, Y)

# (2)
torch.matmul(X, Y)

# (3)
X@Y
```

batch甚至更多都可以, 只要保证最后两个维度匹配就行。
```python
>>> X = torch.randn(2,3,4,5)
>>> Y = torch.randn(2,3,5,6)
>>> (X@Y).shape
torch.Size([2, 3, 4, 6])
```

> 和向量乘就是向量，和矩阵乘就是矩阵

```python
A = torch.rand(3,4)
B = torch.rand(4,1) 
C = torch.rand(4) 

A@B     # [3, 1], tensor([[1.0079],[0.7648],[1.0638]])
A@C     # [3], tensor([0.8935, 1.2776, 0.6071])
```


### other
sum求和，降低维度
```python
Z = torch.ones(3,4)

# 对张量中的所有元素进行求和，会产生一个单元素张量
Z.sum()
# tensor(12.) torch.Size([3, 4])

# 沿着轴求和,即输入轴0的维数在输出形状中消失。
Z.sum(axis=0)
# tensor([3., 3., 3., 3.]) torch.Size([4])
Z.sum(axis=1)
# tensor([4., 4., 4.]) torch.Size([3])

# 沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和。
Z.sum(axis=[0,1])
# tensor(12.)
```

sum，非降维求和
```python
Z.sum(axis=1, keepdims=True)
# tensor([[4.],[4.],[4.]])
```

元素的累积总和，此函数不会沿任何轴降低输入张量的维度
```python
T = torch.arange(12).reshape(3, 4)

# T[i] = T[i-1] + T[i]
T.cumsum(axis=0)
#  tensor([[ 0,  1,  2,  3],
#          [ 4,  6,  8, 10],
#          [12, 15, 18, 21]]),

# T[:, i] = T[:, i-1] + T[;, i]
T.cumsum(axis=1)
#  tensor([[ 0,  1,  3,  6],
#          [ 4,  9, 15, 22],
#          [ 8, 17, 27, 38]]))
```

张量中元素的总数，number of element，即形状的所有元素乘积
```python
X.numel()
```

平均值
```python
# (1) mean: Input dtype must be either a floating point or complex dtype
a = X.float().mean()

# (2) 平均值 = 所有元素求和 / 元素个数
b = X.sum() / X.numel()
```
最大值的维度序号
```python
a = torch.arange(16).reshape(2,8)
torch.argmax(a, dim=0)
torch.argmax(a, dim=1)
```

限定元素大小 clip/clamp [min, max]
```python
input = torch.randint(20, (1, 10))
output = input.clamp(min=0, max=9)
# tensor([[15,  4,  8, 11, 12, 18, 11, 12,  8,  0]])
# tensor([[9, 4, 8, 9, 9, 9, 9, 9, 8, 0]])
```

## 维度操作
如果有4个维度，`dim=-1`, 等同`dim=3`。

> 张量的序列变张量 cat stack
```python
X = torch.ones(3,4)
Y = torch.ones(3,4)

# 按外层的列
a = torch.cat((X, Y), dim=0)    # [6, 4]
# 按内层的列
b = torch.cat((X, Y), dim=1)    # [3, 8]

# 元组，或者列表
d1 = torch.cat([X, Y], dim=0)   # [6, 4]
d2 = torch.cat((X, Y), dim=0)   # [6, 4]
```
```python
X = torch.ones(3,4)
Y = torch.ones(3,4)

# 元组，或者列表
tensor_0 = torch.stack([X, Y], dim=0)    # [2, 3, 4]
tensor_1 = torch.stack([X, Y], dim=1)    # [3, 2, 4]
tensor_2 = torch.stack([X, Y], dim=2)    # [3, 4, 2]

# 提升最后一个维度: RGB图片，前面是xy坐标，最后一个是RGB通道；最后一个是样本特征features
tensor_2 = torch.stack([X, Y], dim=-1)    # [3, 4, 2]
```
> 张量升维，在哪个地方加一个维度 unsqueeze squeeze

`unsqueeze()`函数起升维的作用,参数表示在哪个地方加一个维度。
```python
X=torch.arange(6).reshape(2, 3)
X.unsqueeze(0)      # torch.Size([1, 2, 3])
X.unsqueeze(1)      # torch.Size([2, 1, 3])
X.unsqueeze(2)      # torch.Size([2, 3, 1])
```
降维`squeeze()`
- `dim=None`： 删除一个张量中所有维数为1的维度

  例如，一个维度为 $(A \times 1 \times B \times C \times 1 \times D)$的张量调用squeeze方法后维度变为$(A \times B \times C \times D)$

- 当给定dim参数后，squeeze操作只作用于给定维度。

  如果输入input的形状为$(A \times 1 \times B)$, squeeze(input, 0)不改变这个张量, 但是squeeze(input, 1) 将把这个张量的形状变为$(A \times B)$.
```python
x = torch.zeros(2, 1, 2, 1, 2)
# @dim=None： 删除一个张量中所有维数为1的维度
torch.squeeze(x)        # torch.Size([2, 2, 2])
# 查询维度0, 不是1, 不改变什么
torch.squeeze(x, 0)     # torch.Size([2, 1, 2, 1, 2])
# 查询维度1, 是1, 删除调
torch.squeeze(x, 1)     # torch.Size([2, 2, 1, 2])
```
split: 
https://blog.csdn.net/qq_42518956/article/details/103882579

norm:
https://zhuanlan.zhihu.com/p/260162240

## 逻辑运算符

```python
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# 逻辑运算符
X == Y

# 会广播没有的维度、或者为1的维度
X == 1                  # [3, 4] match []
X == torch.ones(3)      # [3, 4] match [3]
X == torch.ones(4)      # [3, 4] match [4]
X == torch.ones(3, 1)   # [3, 4] match [3]
X == torch.ones(1, 4)   # [3, 4] match [3]
X == torch.ones(3, 4)   # [3, 4] match [3, 4]
X == torch.ones(6)      # [3, 4] cannot match [4]
X == torch.ones(8)      # [3, 4] cannot match [4]
X == torch.ones(6, 4)   # [3, 4] cannot match [6, 4]
```
```python
# Mask 筛选：以下两个结果相反
X = torch.arange(4).reshape(2,2)
M = X > 1     # 想相反，~M

# 矩阵乘法运算快
X * M
# tensor([[0, 0],
#         [2, 3]])

# 索引运算慢
X[M] = 0
# tensor([[0, 1],
#         [0, 0]])
```


## 转换为其他Python对象

```python
# tensor -> numpy
A = torch.tensor(1).numpy()
# Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead. 
# A = torch.tensor(1.0, requires_grad=True).numpy()   # 不行
A = torch.tensor(1.0, requires_grad=True).detach().numpy()  # 需要detach()


# numpy -> tensor
B1 = torch.from_numpy(np.array(1.2))
B2 = torch.tensor(np.array(1.2))

# 此时B是float64，在上面浮点数中有提及。
```

```python
# 将大小为1的张量转换为Python标量
a = torch.tensor([3.5])
a.item(), float(a), int(a)
# 3.5, 3.5, 3
```

## 拷贝

### 拷贝

深拷贝
```python
B = X.clone()  # 通过分配新内存，将A的一个副本分配给B
```
浅拷贝
```python
X = torch.rand(3, 4)
Y = X
# 原来的X也被改变
Y[1:2] = 2
```

## torch.gather / torch.Tensor.gather

> `torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor`

将dim维度的索引改成index, i和j根据是index的shape来

```python
for i in out.shape[0]:
    for j in out.shape[1]:
        out[i][j] = input[index[i][j]][j]  # if dim == 0
        out[i][j] = input[i][index[i][j]]  # if dim == 1

out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

(1) `output.shape = index.shape` 确定最后输出的output的shape必须与index的相同

(2) 对output所有值的索引，按shape方式排出来，也就是`[[(0,0),(0,1),(0,2)]]`

(3) 还是对output，拿index里的值替换上面dim指定位置，dim=0替换行，dim=1即替换列。这里dim=1, 变成`[[(0,2),(0,1),(0,0)]]`

(4) 按这个索引获取tensor_0相应位置的值，填进去就好了，得到torch.tensor([[5,4,3]])

https://zhuanlan.zhihu.com/p/352877584

https://blog.csdn.net/weixin_42200930/article/details/108995776

替换 index 中索引的dim维度为其值。拿着替换后的索引去 input找。
```python
>>> tensor_0 = torch.tensor([4, 3, 5, 7, 6, 8])
>>> tensor_0.gather(dim=0, index=torch.tensor([0, 1, 4]))
tensor([4, 3, 6])
```

```python
>>> tensor_0 = torch.tensor([[ 3,  4,  5], [ 6,  7,  8], [ 9, 10, 11]])			# torch.Size([3, 3])
# 行走：[[2,1,0]]→[[tensor_0(2,0),tensor_0(1,1),tensor_0(0,2)]]
>>> tensor_0.gather(dim=0, index=torch.tensor([[2, 1, 0]]))
tensor([[9, 7, 5]])
>>> tensor_0.gather(dim=1, index=torch.tensor([[2, 1, 0]]))
tensor([[5, 4, 3]])

>>> tensor_0.gather(dim=0, index=torch.tensor([[2], [1], [0]]))
tensor([[9],
        [6],
        [3]])
>>> tensor_0.gather(dim=1, index=torch.tensor([[2], [1], [0]]))
tensor([[5],
        [7],
        [9]])

>>> torch.gather(tensor_0, 0, torch.tensor([[0, 0], [1, 0]]))
tensor([[3, 4],
        [6, 4]])
>>> torch.gather(tensor_0, 1, torch.tensor([[0, 0], [1, 0]]))
tensor([[3, 3],
        [7, 6]])
>>> tensor_0.gather(dim=0, index=torch.tensor([[2, 1], [1, 1], [0, 1]]))
tensor([[9, 7],
        [6, 7],
        [3, 7]])


import torch

# 创建一个三维张量，假设 B=2, N=3, M=4
tensor = torch.tensor([[[ 1,  2,  3,  4],
                        [ 5,  6,  7,  8],
                        [ 9, 10, 11, 12]],

                       [[13, 14, 15, 16],
                        [17, 18, 19, 20],
                        [21, 22, 23, 24]]])

# 创建一个索引张量，假设 B=2, num_indices=2
indices = torch.tensor([[[0, 1, 1],
                        [1, 0, 1],
                        [1, 0, 1]],
                        [[1, 0, 1],
                         [0, 1, 1],
                         [1, 0, 1]]])

# 使用 torch.gather 收集张量
gathered_tensor = torch.gather(tensor, 0, indices)

# 输出结果
print("Original Tensor:")
print(tensor)
print("\nIndices Tensor:")
print(indices)
print("\nGathered Tensor:")
print(gathered_tensor)
```

### 最常用：idx索引

```python
import torch

# 创建一个三维张量，假设 B=2, N=3, M=4
tensor = torch.tensor([[[ 1,  2,  3,  4],
                        [ 5,  6,  7,  8],
                        [ 9, 10, 11, 12]],

                       [[13, 14, 15, 16],
                        [17, 18, 19, 20],
                        [21, 22, 23, 24]]])

#################### 选择不同bathch的数据 [0,1,1] 剩下的两个维度值都是batch维度的同一个值
indices = torch.tensor([0, 1, 1]).view(-1, 1, 1).expand(-1, 3, 4)
'''
tensor([[[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]])
'''
gathered_tensor = torch.gather(tensor, 0, indices)
'''
tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]],
        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]],
        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]])
'''
```
## 内存大小

32位浮点数，32bit，4B
```python
>>> a = torch.rand([16, 16])
>>> a.element_size()            # 字节大小
4
>>> a.nelement()
256
>>> a.element_size() * a.nelement()
1024
```