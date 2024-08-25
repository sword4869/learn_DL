[toc]

```bash
pip install numpy
```
```python
import numpy as np
```

## 性质

```python
arr = np.array([[1., 2., 3.], [4., 5., 6.]])  # 创建ndarray
 
# 元素个数
print(arr.dtype)   # float64
 
# 维度
print(arr.ndim)    # 2
 
# shape
print(arr.shape)   # (2, 3)   两行三列
 
# 元素个数
print(arr.size)    # 6
```

> shape

```python
arr1=np.array(1)
print(arr1)
#1
print(arr1.shape)
#()
 
arr2=np.array([1,2,3])
print(arr2)
#[1 2 3]
print(arr2.shape)
#(3,)
 
arr3=np.array([[1,2,3]])
print(arr3)
#[[1 2 3]]
print(arr3.shape)
#(1,3)
 
arr4=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr4)
'''
[[[ 1  2  3]
  [ 4  5  6]]
 [[ 7  8  9]
  [10 11 12]]]
'''
print(arr4.shape)
#(2,2,3)
```

### print小数

```python
np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True, precision=4)
```

## 创建ndarray

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzI2MDQyMjkxX2Qza1cyTllOYnMuanBn?x-oss-process=image/format,png)

### np.array()

```python
import numpy as np
 
# 一维
a = {1,2,3}
data1 = np.array(a)
print(data1)
print(type(data1))
'''
{1, 2, 3}
<class 'numpy.ndarray'>
'''
 
b = [4,5,6]
data2 = np.array(b)
print(data2)
print(type(data2))
'''
[4 5 6]
<class 'numpy.ndarray'>
'''
 
# 多维
c = [[1,2,3],[4,5,6]]
data3 = np.array(c)
print(data3)
print(type(data3))
'''
[[1 2 3]
[4 5 6]]
<class 'numpy.ndarray'>
'''
 
d = {{1,2,3},{4,5,6}}
data4 = np.array(d)
print(data4)
print(type(data4))
# TypeError: unhashable type: 'set'

# 所以array()中的数据最好写成列表类型
```

### np.zeros() 、ones() 、empty()

zeros和ones分别可以创建指定长度或形状的全0或全1数组，empty可以创建一个没有任何具体值的数组。

> 函数原型:

```python
np.zeros(shape,dtype=float,order='C')
np.ones(shape,dtype=float,order='C')
np.empty(shape,dtype=float,order='C')
```

> 要用这些方法创建多维数组，只需传入一个表示形状的元组类型即可: 

```python
print(np.zeros(4))
'''
[ 0.  0.  0.  0.]
'''
 
print(np.zeros((3,4)))
#注意是元组类型，(3,4)是传给shape的
'''
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
'''
 
print(np.zeros((2,3,4)))
#从高维到低维
'''
[[[ 0.  0.  0.  0.]
  [ 0.  0.  0.  0.]
  [ 0.  0.  0.  0.]]
 [[ 0.  0.  0.  0.]
  [ 0.  0.  0.  0.]
  [ 0.  0.  0.  0.]]]
'''
```

> 注意：认为np.empty会返回全0数组的想法是不安全的。只分配内存空间，不进行初始化，它返回的都是一些未初始化的垃圾值。 

```python
print(np.empty((2,3)))
'''
[[ 2.1096553  -0.09990896  1.04218815]
 [-0.13491275 -0.29338706 -0.90817572]]
'''
```

### np.arange()

arange是Python内置函数range的数组版，产生从0~N-1的整数ndarray。注意：不到N

> 一个参数的arange() 

```python
print(np.arange(8))
#[0 1 2 3 4 5 6 7]
```

> 两个参数的arange() 

```python
arr=np.arange(1,10)
print(arr)
#[1 2 3 4 5 6 7 8 9]
```

>  三个参数的arange()

```python
arr=np.arange(1,2,0.1)
print(arr)
#[ 1.   1.1  1.2  1.3  1.4  1.5  1.6  1.7  1.8  1.9]
```

> 产生特定shape的ndarry数组 

```python
arr=np.arange(28).reshape((3,2,4))
#reshape()里传入一个元组类型
```

## ndarray的数据类型

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzI2MjU4OTA0X3pITzZOZjlicUsuanBn?x-oss-process=image/format,png)

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzI2MjczMzczX0tFYTdkaXJJZ2guanBn?x-oss-process=image/format,png)

> 通常只需要知道你所处理的数据的大致类型是浮点数、复数、整数、布尔值、字符串，还是普通的Python对象即可 

### 指定类型： 

```python
arr1=np.array([1,2,3])
print(arr1.dtype)
#int32
#自动选择合适的类型
 
arr2=np.array([4,5,6],dtype=np.int64)
print(arr2.dtype)
#int64
```

```python
#简洁形式
empty_uint32 = np.empty(8, dtype='u4')
```

### 转化类型：

> astype()

```python
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
print(arr)
print(arr.astype(np.int32))
'''
[  3.7  -1.2  -2.6   0.5  12.9  10.1]
[ 3 -1 -2  0 12 10]
'''
#整数被转换成了浮点数。如果将浮点数转换成整数，则小数部分将会被截取删除：
```

> 如果某字符串数组表示的全是数字，也可以用astype将其转换为数值形式。
> 
> 注意：使用numpy.string\_类型时，一定要小心，因为NumPy的字符串数据是大小固定的，发生截取时，不会发出警告。

```python
arr= np.array(['1.25', '-9.6', '42.00000002344'], dtype=np.string_)
print(arr.astype(np.float))
#[  1.25        -9.6         42.00000002]
```

### np.array()和np.asarray()的区别

`array` 和 `asarray` 都可以将结构数据转化为 `ndarray` 
- 当数据源是列表时，都会copy出一个副本，占用新的内存。
- 当数据源是 `ndarray` 时, `array` 仍然会copy出一个副本，占用新的内存，但 `asarray` 不会，从而修改原np数组，也会变化。
- 混合的情况`[array([1, 1, 1]), array([1, 2, 1]), array([1, 1, 1])]`，同列表
```python
import numpy as np
 
# example 1: 列表
data1 = [[1,1,1],[1,1,1],[1,1,1]]
arr2 = np.array(data1)
arr3 = np.asarray(data1)
data1[1][1] = 2
print(data1[1][1], arr2[1][1], arr3[1][1])

import numpy as np
 
# example 2: ndarry
arr4 = np.ones((3,3))
arr5 = np.array(arr4)
arr6 = np.asarray(arr4)
arr4[1][1] = 2
print(arr4[1][1], arr5[1][1], arr6[1][1])

# example 3: 混合，[ndarry]
arr7 = [np.array([1,1,1]) for i in range(3)]
arr8 = np.array(arr7)
arr9 = np.asarray(arr7)
arr7[1][1] = 2
print(arr7[1][1], arr8[1][1], arr9[1][1])

'''
2 1 1
2.0 1.0 2.0
2 1 1
'''
```
## 还原list

```python
l = arr.tolist()
# [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
```
## ndarray数组的运算

> 数组与标量的算术运算会将标量值传播到各个元素：

```python
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr+1)
print(arr*2)
print(1/arr)
print(arr**2)
'''
[[ 2.  3.  4.]
 [ 5.  6.  7.]]
[[  2.   4.   6.]
 [  8.  10.  12.]]
[[ 1.          0.5         0.33333333]
 [ 0.25        0.2         0.16666667]]
[[  1.   4.   9.]
 [ 16.  25.  36.]]
'''
```

> 数组与数组之间的算术运算 

```python
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[2., 3., 4.], [5., 6., 7.]])
print(arr2 - arr1)
print(arr1 * arr2)
'''
[[ 1.  1.  1.]
 [ 1.  1.  1.]]
[[  2.   6.  12.]
 [ 20.  30.  42.]]
'''
#arr*arr并非是矩阵乘法，而是对应元素之间的数字相乘
```

> 数组之间的比较会生成布尔值数组：

```python
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[2., 3., 4.], [5., 6., 7.]])
print(arr2 > arr1)
print(arr2 == arr1)
'''
[[ True  True  True]
 [ True  True  True]]
[[False False False]
 [False False False]]
'''
#>、<、==、>=、<=.注意==
```
## 基本的切片与索引

### 基本的切片和索引

注意：当你将一个标量值赋值给一个切片时（如arr[5:8]=12），该值会自动传播到整个选区。

```python
import numpy as np 
arr = np.arange(10)
 
print (arr)
#[0 1 2 3 4 5 6 7 8 9]
print (arr[5])
#5
print (arr[0:2])
#[0 1]
arr[0:2] = 12
print (arr)
#[12 12  2  3  4  5  6  7  8  9]
```

```python
#与内置数组的区别
arr1=[0,1,2,3,4,5]
arr1[0:2]=[6]
print(arr1[0:2])
#[6,2]
print(arr1)
#[6,2,3,4,5]
```

```python
#ps
print (arr)
#[0 1 2 3 4 5 6 7 8 9]
slice=arr[0:4]
print(slice)
#[0 1 2 3]
slice=666
print(arr)
#[0 1 2 3 4 5 6 7 8 9]
```

```python
#ps
print (arr)
#[0 1 2 3 4 5 6 7 8 9]
slice=arr[0:4]
print(slice)
#[0 1 2 3]
slice[:]=666
print(arr)
```

### 多维数组的索引

> 在多维数组中，如果省略了后面的索引，则返回对象会是一个维度低一点的ndarray
> 
> 如：在一个二维数组中，各索引位置上的元素不再是标量而是一维数组： 

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2])
#[7 8 9]
```

>  选取单个元素

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#同C/C++的风格
print(arr2d[0][2])
#3
#自己的风格，列表
print(arr2d[0,2])
#3
```

> axis值 

 ![](https://img-blog.csdnimg.cn/20190320193647802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbmRhbHBob240ODY5,size_16,color_FFFFFF,t_70)



## 花式切片

![](https://img-blog.csdnimg.cn/20200615133536252.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbmRhbHBob240ODY5,size_16,color_FFFFFF,t_70)

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 
#[:2]一个切片值，表示行切片，从0不到2行
print(arr2d[:2])
'''
[[1 2 3]
 [4 5 6]]
'''
 
#两个切片值，[行切片，列切片]
print(arr2d[:2,1:])
'''
[[2 3]
 [5 6]]
'''
 
#选取第二行的前两列
print(arr2d[1,:2])
#[4 5]
#选区前两行的第二列
print(arr2d[:2,1])
#[2 5]
 
#选取前两列
print(arr2d[:,:2])
'''
[[1 2]
 [4 5]
 [7 8]]
'''
```

## 布尔型索引 

通过布尔型索引选取数组中的数据，将总是创建数据的副本，即使返回一模一样的数组也是如此。 

> 对names和字符串"Bob"的比较运算将会产生一个布尔型数组：

```python
names=np.array(['A','B','C','A'])
print(names=='A')
#[ True False False True]
```

> 布尔型数组可用于数组索引
> 
> 注意：布尔型数组的长度必须跟被索引的轴长度一致

```python
arr=np.empty((4,3))
print(arr)
'''
[[  1.16337425e-311   2.81617418e-322   0.00000000e+000]
 [  0.00000000e+000   1.69119873e-306   8.60952352e-072]
 [  4.46442178e-090   1.05118281e-046   1.55029986e+184]
 [  2.14704562e+184   6.48224659e+170   4.93432906e+257]]
'''
print(arr[names=='A'])
'''
[[  1.16337425e-311   2.81617418e-322   0.00000000e+000]
 [  2.14704562e+184   6.48224659e+170   4.93432906e+257]]
'''
#相当于arr[0]、arr[3]
```

> 布尔型数组跟切片、整数混合使用：

```python
print(arr[names=='A',:2])
'''
[[  1.16337425e-311   2.81617418e-322]
 [  2.14704562e+184   6.48224659e+170]]
'''
```

> ~操作符用来反转布尔条件很好用： 

```python
condition=names=='A'
print(arr[~condition])
'''
[[  0.00000000e+000   1.69119873e-306   8.60952352e-072]
 [  4.46442178e-090   1.05118281e-046   1.55029986e+184]]
'''
```

>  组合应用多个布尔条件，使用&（和）、|（或）的布尔算术运算符
> 
> 注意：组合条件要用括号括起来
> 
> 注意：Python关键字and和or在布尔型数组中无效。要使用&与|。

```python
print(arr[names=='A'|names=='B'])
#error
print(arr[(names=='A')|(names=='B')])
'''
[[  1.16337425e-311   2.81617418e-322   0.00000000e+000]
 [  0.00000000e+000   1.69119873e-306   8.60952352e-072]
 [  2.14704562e+184   6.48224659e+170   4.93432906e+257]]
'''
```

> 通过布尔型数组设置值

```python
arr=np.array([[-1,-2,-3],[-1,0,1],[1,2,3]])
print(arr)
arr[arr<0]=0
print(arr)
'''
[[-1 -2 -3]
 [-1  0  1]
 [ 1  2  3]]
[[0 0 0]
 [0 0 1]
 [1 2 3]]
'''
#第二行中还有'1'，说明是设置单个值为0，而不是设置整行整列为0
```

```python
arr=np.array([[-1,-2,-3],[-1,0,1],[1,2,3]])
print(arr)
tags=np.array([True,True,False])
arr[tags]=0
print(arr)
'''
[[-1 -2 -3]
 [-1  0  1]
 [ 1  2  3]]
[[0 0 0]
 [0 0 0]
 [1 2 3]]
'''
#设置整行整列为0
```

## 花式索引

花式索引跟切片不一样，它总是将数据复制到新数组中。

> 括号的`[]`的个数，一个是基本的索引，两个及以上的才是花式索引，并且比其少的括号和其一样，比其多的括号就多。

所以，我们就选用两个括号就行，`[[传入的数组]]`

```python
arr=np.arange(36).reshape((3,3,4))
print(arr)
'''
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]
 [[24 25 26 27]
  [28 29 30 31]
  [32 33 34 35]]]
'''

# 只有一个[]的数组是基本的索引
print(arr[2,0,1])
# 25


# 花式索引传入的数组用两个以上的[]
print(arr[[2,0]])
# 就是 [arr[2], arr[0]]
'''
[[[24 25 26 27]
  [28 29 30 31]
  [32 33 34 35]]
 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]]
'''
print(arr[[[2,0]]])
'''
[[[24 25 26 27]
  [28 29 30 31]
  [32 33 34 35]]
 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]]
'''
print(arr[[[[[[2,0]]]]]])
'''
[[[[[[24 25 26 27]
     [28 29 30 31]
     [32 33 34 35]]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]]]]]
 '''
```

> 使用负数索引将会从末尾开始选取行 

```python
print(arr[[-1,-2,-3]])
'''
[[[24 25 26 27]
  [28 29 30 31]
  [32 33 34 35]]
 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]
 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]]
 '''
```

>  一次传入多个索引数组会有一点特别。它返回的是一个一维数组，其中的元素对应各个索引元组。
> 
> 注意：对应，那么就要求传入的数组的shape一样。

```python
arr = np.arange(16).reshape((4, 4))
print(arr)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
'''
print(arr[[3,1,2,0], [0, 3, 1, 2]])
# [12  7  9  2]
# 就是arr[3,0],[1,3],[2,1],[0,2]
print(arr[[3,1,2,0],[2,3]])
# error,不对应
```

```python
arr=np.arange(36).reshape(3,3,4)
print(arr[[2,1,0],[1,0,2]])
'''
[[28 29 30 31]
 [12 13 14 15]
 [ 8  9 10 11]]
'''
print(arr[[2,1,0],[1,0,2],[3,2,0]])
#[31 14  8]
```

> 也可以是ndarry, 相当于把`[[]]`的内括号移动到`np.array([])`中了。

```python
arr=np.arange(36).reshape((3,3,4))
print(arr[np.array([2,0])])
'''
[[[24 25 26 27]
  [28 29 30 31]
  [32 33 34 35]]
 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]]
'''
```


> 花式索引加切片索引等：但是不能`arr[[1, 2], [:3 , :3]]`，所以
> 
> 就是在花式索引 `arr[[传入数组]]` 后面加其他的操作，如切片索引`[: , :3]`。就是`arr[[]][: , :3]`。之间的空格是为了看起来方便，没有也行

```python
arr = np.arange(16).reshape((4, 4))
print(arr)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
'''
print(arr[[2,1,0,3]] [:,:3])
'''
[[ 8  9 10]
 [ 4  5  6]
 [ 0  1  2]
 [12 13 14]]
'''
```

## 数组转置

转置是重塑的一种特殊形式，它返回的是源数据的视图（不会进行任何复制操作）

转置方法有`transpose()`、`.T`和`swapaxes()`.

关系：在`transpose()`的基础上分化出`.T`和`swapaxes()`。

ps:transpose是转置的意思，pose是姿势的意思。swap是交换，axes通axis，axis是轴的意思。

> transpose（1,0,2）：表示将（[0], [1], [2]）转换为（[1], [0], [2]）。简单理解就是，将不同位置元素替换掉。
> 
> 比如：arr[0, 0, 0]，第一位和第二位转换后，仍是arr[0, 0, 0]。arr[0 , 1, 0] = 4, 转换后为 arr[1, 0, 0] = 8。同理arr[1, 0 , 0]转换为 arr[0, 1, 0]。此次类推。

```python
arr = np.arange(16).reshape((2, 2, 4))
print(arr)
'''
[[[ 0  1  2  3]
  [ 4  5  6  7]]
 [[ 8  9 10 11]
  [12 13 14 15]]]
'''
print(arr.transpose(1,0,2))
'''
[[[ 0  1  2  3]
  [ 8  9 10 11]]
 [[ 4  5  6  7]
  [12 13 14 15]]]
'''
```

> T转置：表示整个顺序颠倒，（[0], [1], [2]）转换为（[2], [1], [0]）。就是内容替换。  

```python
arr = np.arange(16).reshape((2, 2, 4))
print(arr.T)
'''
[[[ 0  8]
  [ 4 12]]
 [[ 1  9]
  [ 5 13]]
 [[ 2 10]
  [ 6 14]]
 [[ 3 11]
  [ 7 15]]]
'''
print(arr.transpose(2,1,0))
'''
[[[ 0  8]
  [ 4 12]]
 [[ 1  9]
  [ 5 13]]
 [[ 2 10]
  [ 6 14]]
 [[ 3 11]
  [ 7 15]]]
'''
```

> T转置适用于一维、二维数组 

```python
onearr=np.array([[1,2]])
print(onearr.T)
'''
[[1]
 [2]]
'''
 
arr = np.arange(12).reshape((3, 4))
print(arr)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
print(arr.T)
'''
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]
'''
```

> swapaxes方法：表示，将其中两个轴互换

```python
arr = np.arange(16).reshape((2, 2, 4))
print(arr.swapaxes(1,2))
'''
[[[ 0  4]
  [ 1  5]
  [ 2  6]
  [ 3  7]]
 [[ 8 12]
  [ 9 13]
  [10 14]
  [11 15]]]
'''
print(arr.transpose(0,2,1))
'''
[[[ 0  4]
  [ 1  5]
  [ 2  6]
  [ 3  7]]
 [[ 8 12]
  [ 9 13]
  [10 14]
  [11 15]]]
'''
```

### torch

`permute()`、`.T`, `transpose()`
numpy到torch
- 核心：`transpose()` → `permute()`
- `swapaxes()` → `transpose()`
- `.T`不变

> permute
```python
A = torch.arange(12).reshape(3, 4)

A.permute(1, 0)

X = torch.ones(B, H, W, C)
# X: [B, H, W, C]，变换成Y: [B, C, H, W]
X.permute(0, 3, 1, 2)
```

> `.T` 和 `transpose()`
```python
A = torch.arange(12).reshape(3, 4)

# 转置
A.T

# [4, 3]
A.transpose(0, 1) == A.transpose(1, 0)
```



## 如何判断两个numpy数组是否相等

- `(array1 == array2)` 返回两个矩阵中对应元素是否相等的逻辑值
- `(array1 == array2).all()`当两个矩阵所有对应元素相等，返回一个逻辑值True
- `(array1 == array2).any()`当两个矩阵所有任何一个对应元素相等，返回一个逻辑值True

## copy

### 引用
```python
a = np.arange(12)
c = a                            
print('c是a吗？', c is a)
print('c是以a为基础建立的吗', c.base is a)
c.shape = 3, 4
print('a的shape', a.shape)    # a的shape会跟着变
c[0] = -1
print('a的值', a[0])          # a的值会跟着变
'''
c是a吗？ True
c是以a为基础建立的吗 False  ## c都已经是a....
a的shape (3, 4)
a的值 [-1 -1 -1 -1]
'''
```



### 视图（View）是浅复制
view方法创建一个与原来数组相同的新对象：
- 可以分享相同的数据 value. 数据不会被复制，修改切片，变动也会体现在原始数组中。
- 唯一可以变的是 shape.
```python
a = np.arange(12)
c = a.view()                     
print('c是a吗？', c is a)
print('c是以a为基础建立的吗', c.base is a)
c.shape = 2, 2
print('a的shape', a.shape)    # a的shape不跟着变
c[0] = -1
print('a的值', a[0])          # 但a的值会跟着变
c[:] = 666
print('a的值', a[0])          # 但a的值会跟着变
'''
c是a吗？ False
c是以a为基础建立的吗 True
a的shape (12,)
a的值 -1
a的值 666
'''
```

数组切片是原始数组的视图。

```python
a = np.arange(12)
c = a[0:4]                       
print('c是a吗？', c is a)
print('c是以a为基础建立的吗', c.base is a)
c.shape = 2, 2
print('a的shape', a.shape)    # a的shape不跟着变
c[0] = -1
print('a的值', a[0])          # 但a的值会跟着变
c[:] = 666
print('a的值', a[0])          # 但a的值会跟着变
'''
c是a吗？ False
c是以a为基础建立的吗 True
a的shape (12,)
a的值 -1
a的值 666
'''
```
### 深复制（Deep Copy）
这个时候d是a的复制，只是单纯的复制，两者没有一点关系：
```python
a = np.arange(12)
c = a.copy()                            
print('c是a吗？', c is a)
print('c是以a为基础建立的吗', c.base is a)
c.shape = 3, 4
print('a的shape', a.shape)    # a的shape不跟着变
c[0] = -1
print('a的值', a[0])          # a的值不会跟着变
'''
是a吗？ False
c是以a为基础建立的吗 False
a的shape (12,)
a的值 0
'''
```