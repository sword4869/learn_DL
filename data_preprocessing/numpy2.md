- [1. 通用函数：快速的元素级数组函数](#1-通用函数快速的元素级数组函数)
  - [1.1. 一元ufunc](#11-一元ufunc)
  - [1.2. 二元ufunc](#12-二元ufunc)
- [2. 可以返回多个数组的ufunc](#2-可以返回多个数组的ufunc)
- [3. out可选参数](#3-out可选参数)
- [4. 将条件逻辑表述为数组运算np.where()](#4-将条件逻辑表述为数组运算npwhere)
- [5. 数学和统计方法](#5-数学和统计方法)
  - [5.1. 聚合计算aggregation](#51-聚合计算aggregation)
  - [5.2. max](#52-max)
  - [5.3. 非聚合](#53-非聚合)
- [6. 对布尔型数组中的True值计数](#6-对布尔型数组中的true值计数)
- [7. 排序](#7-排序)
  - [7.1. 三种排序算法](#71-三种排序算法)
  - [7.2. numpy.sort()](#72-numpysort)
  - [7.3. numpy.argsort()](#73-numpyargsort)
  - [7.4. numpy.lexsort()](#74-numpylexsort)
- [8. 唯一化以及其它的集合逻辑 ](#8-唯一化以及其它的集合逻辑)
- [9. 用于数组的文件输入输出](#9-用于数组的文件输入输出)
- [10. 线性代数](#10-线性代数)
- [11. 伪随机数生成 np.random](#11-伪随机数生成-nprandom)
  - [11.1. seed()](#111-seed)
  - [11.2. rand()均匀分布](#112-rand均匀分布)
  - [11.3. randint()](#113-randint)
  - [11.4. randn()正态分布](#114-randn正态分布)
- [12. 一个简单的随机漫步](#12-一个简单的随机漫步)
- [13. 通过内置的random模块以纯Python的方式](#13-通过内置的random模块以纯python的方式)
- [14. 用np.random模块 ](#14-用nprandom模块)
- [15. 一次模拟多个随机漫步](#15-一次模拟多个随机漫步)


## 1. 通用函数：快速的元素级数组函数

通用函数（即ufunc）是一种对ndarray中的数据执行元素级运算的函数。

### 1.1. 一元ufunc

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzM1NDQ1MTg3XzBCS1pabTFsSjAuanBn?x-oss-process=image/format,png)

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzM1NDU4NjUyX1JGU2hOMGwyeHYuanBn?x-oss-process=image/format,png)

> sqrt 

```
arr = np.arange(10)
print(arr)
#[0 1 2 3 4 5 6 7 8 9]
print(np.sqrt(arr))
'''
[ 0.          1.          1.41421356  1.73205081  2.          2.23606798
  2.44948974  2.64575131  2.82842712  3.        ]
'''
print(np.exp(arr))
'''
[  1.00000000e+00   2.71828183e+00   7.38905610e+00   2.00855369e+01
   5.45981500e+01   1.48413159e+02   4.03428793e+02   1.09663316e+03
   2.98095799e+03   8.10308393e+03]
'''
```

### 1.2. 二元ufunc

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzM1NDcwNDA4XzhGYmVtZ0JtamIuanBn?x-oss-process=image/format,png)

> maximum 

```
x = np.random.randn(4)
y = np.random.randn(4)
print(x)
print(y)
print(np.maximum(x, y))
'''
[ 0.10616977 -0.49212002  0.99288887  0.01832999]
[-0.43932326 -0.49726355  0.72992509  0.05767231]
[ 0.10616977 -0.49212002  0.99288887  0.05767231]
'''
```

## 2. 可以返回多个数组的ufunc

> modf就是一个例子，它是Python内置函数divmod的矢量化版本，它会返回浮点数数组的小数部分（第一个返回的数组）和整数部分（第二个）： 

```
arr = np.random.randn(4) * 9
print(arr)
remainder, whole_part = np.modf(arr)
print(remainder)
print(whole_part)
'''
[  9.72951779 -11.1838008   -5.79476731   8.58079314]
[ 0.72951779 -0.1838008  -0.79476731  0.58079314]
[  9. -11.  -5.   8.]
'''
```

## 3. out可选参数

> Ufuncs可以接受一个out可选参数，这样就能在数组原地进行操作： 

```
arr=np.random.randn(4)
print(arr)
# [-0.21348361 -0.81417907  1.03700407  0.72705193]
 
#没有out可选参数，原变量不动
print(np.sqrt(arr))
# [       nan        nan 1.01833397 0.8526734 ]
print(arr)
# [-0.21348361 -0.81417907  1.03700407  0.72705193]
 
#有out可选参数，原变量改动
print(np.sqrt(arr, arr))
# [       nan        nan 1.01833397 0.8526734 ]
print(arr)
# [       nan        nan 1.01833397 0.8526734 ]
```


## 4. 将条件逻辑表述为数组运算np.where()

> numpy.where函数是三元表达式x if condition else y的矢量化版本,更快，更简洁
> 
> 参数：
> 
> np.where的第一个参数是表述条件逻辑的数组
> 
> np.where的第二个和第三个参数不必是数组，它们都可以是标量值。

```
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = np.where(cond, xarr, yarr)
print(result)
#[ 1.1  2.2  1.3  1.4  2.5]
```

> 假设有一个由随机数据组成的矩阵，你希望将所有正值替换为2，将所有负值替换为－2。若利用np.where，则会非常简单： 

```
arr = np.random.randn(4, 4)
print(arr)
print(arr > 0)
print(np.where(arr > 0, 2, -2))
'''
[[ 0.447964    0.58736384 -1.14557629 -1.13452156]
 [-0.9571849   0.19986246  1.50479059 -0.59422043]
 [ 0.61614284  0.99910225  0.50868714 -0.68930212]
 [ 2.49150798  0.10075012  0.27746326 -0.54341567]]
[[ True  True False False]
 [False  True  True False]
 [ True  True  True False]
 [ True  True  True False]]
[[ 2  2 -2 -2]
 [-2  2  2 -2]
 [ 2  2  2 -2]
 [ 2  2  2 -2]]
'''
```

>  使用np.where，可以将标量和数组结合起来。例如，我可用常数2替换arr中所有正的值

```
print(np.where(arr > 0, 2, arr))
'''
[[ 0.60592778 -1.12237571 -0.62943517  0.63504135]
 [-0.17111278  1.13233184  0.57757454  1.3412847 ]
 [ 1.19106342  0.42438377  0.24304122 -1.6544575 ]
 [-0.12271689  0.16447323 -1.78730423  0.47386953]]
[[ 2.         -1.12237571 -0.62943517  2.        ]
 [-0.17111278  2.          2.          2.        ]
 [ 2.          2.          2.         -1.6544575 ]
 [-0.12271689  2.         -1.78730423  2.        ]]
'''
```

## 5. 数学和统计方法

 ![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzM3Mzk5NDE0X3pzVkVjOU0zdWouanBn?x-oss-process=image/format,png)

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzM3NDI3NTc5X3hVTTA3ZTJQYWIuanBn?x-oss-process=image/format,png)

![](https://img-blog.csdnimg.cn/20200603223655496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbmRhbHBob240ODY5,size_16,color_FFFFFF,t_70)

```
a = np.array([1,2])
b = np.array([3,4])
 
"""
注意这两个是不一样的
"""
# 方差是减去平均值
np.var(a - b)
# 0.0
 
# 均方误差是减去另一组的每个数
np.mean((a - b) ** 2)
# 4.0
```

### 5.1. 聚合计算aggregation

可以通过数组上的一组数学函数对整个数组或某个轴向的数据进行统计计算。sum、mean以及标准差std等聚合计算（aggregation，通常叫做约简（reduction））

> 既可以当做数组的实例方法调用，也可以当做顶级NumPy函数使用 

```
arr = np.random.randn(5, 4)
print(arr)
print(arr.mean())
print(np.mean(arr))
'''
[[ 0.0076106  -1.69802614  0.17551021 -0.92825912]
 [ 0.0429896   0.41676059  0.75423905 -2.00118662]
 [ 2.2831568   0.06876217  0.06897961  1.32156444]
 [ 0.05282213 -0.71710608  0.49313807  0.81165569]
 [ 0.46388897  1.99487123 -0.7220173   0.879218  ]]
0.188428595892
0.188428595892
'''
```

>  axis可选项参数：mean和sum这类的函数就变成计算**该轴向上**的统计值，最终结果是一个少一维的数组

![](https://img-blog.csdnimg.cn/20190320193647802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbmRhbHBob240ODY5,size_16,color_FFFFFF,t_70)

```
arr = np.arange(4).reshape((2,2))
print(arr)
print(arr.mean(axis=1))    #行，计算行的平均值
print(arr.sum(axis=0))     #列，计算每列的和
'''
[[0 1]
 [2 3]]
[ 0.5  2.5]
[2 4]
'''
```

### 5.2. max

由于`1<nan`, `1>nan``1==nan` 均为 `False`
- `np.max(a, axis)`：有nan则最大值返回nan
- `np.nanmax(a, axis)`：排除nan
- `np.maximum(a, b, axis)`：这才是两个数组。还是有nan问题。

### 5.3. 非聚合

> cumsum和cumprod之类的方法则不聚合，而是产生一个由中间结果组成的数组

```
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
print(arr.cumsum())
#[ 0  1  3  6 10 15 21 28]
```

> axis可选参数与高维数组

```
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(arr)
print(arr.cumsum(axis=0))        #列，计算每列的累计和
print(arr.cumprod(axis=1))       #行，计算每行的累计和
'''
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[ 0  1  2]
 [ 3  5  7]
 [ 9 12 15]]
[[  0   0   0]
 [  3  12  60]
 [  6  42 336]]
'''
```

## 6. 对布尔型数组中的True值计数

> sum经常被用来对布尔型数组中的True值计数

```
import numpy as np
arr=np.random.randn(5)
print(arr)
print((arr>0).sum())
'''
[-0.86757426 -0.48756643  0.18361648  0.54542627 -0.63947586]
2
'''
```

>  any用于测试布尔数组中是否存在一个或多个True，而 all则检查布尔数组中所有值是否都是True。
> 
> 这两个方法也能用于非布尔型数组，所有非0元素将会被当做True。

```
bools = np.array([False, False, True, False])
print(bools.any())
print(bools.all())
```

```
import numpy as np
arr=np.random.randn(5)
print(arr)
print((arr>0).any())
print((arr>0).all())
'''
[-0.4116666  -0.20068141  1.57812722  0.04348732 -0.82217497]
True
False
'''
```

## 7. 排序

###  7.1. 三种排序算法

| 种类 | 速度 | 最坏情况 | 工作空间 | 稳定性 |
| --- | --- | --- | --- | --- |
| 'quicksort'(快速排序) | 1 | O(n^2) | 0 | 否 |
| 'mergesort'(归并排序) | 2 | O(n\*log(n)) | ~n/2 | 是 |
| 'he**a**psort'(堆排序) | 3 | O(n\*log(n)) | 0 | 否 |

### 7.2. numpy.sort()

sort()函数返回输入数组的排序副本。 它有以下参数：

```
numpy.sort(a, axis, kind, order)
```

其中：

| 序号 | 参数及描述 |
| --- | --- |
| 1 | a 要排序的数组 |
| 2 | axis 沿着它排序数组的轴，如果没有数组会被展开，沿着最后的轴排序。默认axis=1 |
| 3 | kind 默认为'quicksort'(快速排序) |
| 4 | order 如果数组包含字段，则是要排序的字段 |

 例子：

```
a = np.array([[3,7],[9,1]])  
print ('我们的数组是：')
print (a) 
print ('\n')  
print ('调用 sort() 函数：')  
print (np.sort(a))  
print ('\n')  
print ('沿轴 0 排序：')  
print (np.sort(a, axis =  0))  
'''
我们的数组是：
[[3 7]
 [9 1]]
调用 sort() 函数：
[[3 7]
 [1 9]]
沿轴 0 排序：
[[3 1]
 [9 7]]
'''
```

### 7.3. numpy.argsort()

numpy.argsort()函数对输入数组沿给定轴执行间接排序，并使用指定排序类型返回数据的索引数组。 这个索引数组用于构造排序后的数组。 

```
x = np.array([3,  1,  2])  
print ('我们的数组是：')  
print (x) 
print ('\n')  
print ('对 x 调用 argsort() 函数：' )
y = np.argsort(x)  
print (y) 
print ('\n')  
print ('以排序后的顺序重构原数组：')  
print (x[y])  
print ('\n')  
print ('使用循环重构原数组：')  
for i in y:  
    print (x[i])
'''
我们的数组是：
[3 1 2]
对 x 调用 argsort() 函数：
[1 2 0]
以排序后的顺序重构原数组：
[1 2 3]
使用循环重构原数组：
1
2
3
'''
```

### 7.4. numpy.lexsort()

函数使用键序列执行间接排序。 键可以看作是电子表格中的一列。 该函数返回一个索引数组，使用它可以获得排序数据。 注意，最后一个键恰好是 sort 的主键。 

## 8. 唯一化以及其它的集合逻辑 

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzQ0NzIzODU5XzRoNkdOYnhTWDAuanBn?x-oss-process=image/format,png)

> np.unique，它用于找出数组中的唯一值并返回已排序的结果： 

```
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(np.unique(names))
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
print(np.unique(ints))
'''
['Bob' 'Joe' 'Will']
[1 2 3 4]
'''
```

>  np.in1d用于测试一个数组中的值在另一个数组中的成员资格，返回一个布尔型数组

```
values = np.array([6, 0, 0, 3, 2, 5, 6])
print(np.in1d(values, [2, 3, 6]))
#[ True False False  True  True False  True]
```

## 9. 用于数组的文件输入输出

NumPy能够读写磁盘上的文本数据或二进制数据。

> np.save将数组数据写入磁盘：
> 
> 默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中的。如果文件路径末尾没有扩展名.npy，则该扩展名会被自动加上。

```
arr = np.arange(10)
np.save('some_array', arr)
```

> np.load读取磁盘上的数组 

```
print(np.load('some_array.npy'))
#[0 1 2 3 4 5 6 7 8 9]
```

>  np.savez可以将多个数组保存到一个未压缩文件中，将数组以关键字参数的形式传入即可

```
np.savez('array_archive.npz', a=arr, b=arr)
arch = np.load('array_archive.npz')
print(arch['b'])
#[0 1 2 3 4 5 6 7 8 9]
```

> 如果要将数据压缩，可以使用numpy.savez\_compressed

```
np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)
```

## 10. 线性代数

>  np.dot计算矩阵内积（点乘积）

```
arr1=np.array([[1],[2]])
arr2=np.array([[3,4]])
print(np.dot(arr1,arr2))
'''
[[3 4]
 [6 8]]
'''
```

```
print(np.dot(arr1, arr2))
'''
[[3 4]
 [6 8]]
'''
```

```
#@符（类似Python 3.5）也可以用作中缀运算符，进行矩阵乘法：
arr1 = np.array([[0.05,0.1],[0.5,0.2]])
arr2 = np.array([[0.05,0.1],[0.5,0.2]])
print(arr1@arr2)
'''
[[ 0.0525  0.025 ]
 [ 0.125   0.09  ]]
'''
```

> numpy.linalg中有一组标准的矩阵分解运算以及诸如求逆和行列式之类的东西。它们跟MATLAB和R等语言所使用的是相同的行业标准线性代数库，如BLAS、LAPACK、Intel MKL（Math Kernel Library，可能有，取决于你的NumPy版本）等：

![](https://img-blog.csdnimg.cn/20190324183349657.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbmRhbHBob240ODY5,size_16,color_FFFFFF,t_70)

```
import numpy as np
arr = np.array([[0.05,0.1],[0.5,0.2]])
print(np.linalg.inv(arr))
'''
[[ -5.     2.5 ]
 [ 12.5   -1.25]]
'''
#print(np.inv(arr)) 报错
```

## 11. 伪随机数生成 np.random

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzk2MDk4OTIxX25QTzMyWVFTUHouanBn?x-oss-process=image/format,png)

![GitHub](https://imgconvert.csdnimg.cn/aHR0cDovL2FsaXl1bnRpYW5jaGlwdWJsaWMuY24taGFuZ3pob3Uub3NzLXB1Yi5hbGl5dW4taW5jLmNvbS9wdWJsaWMvZmlsZXMvaW1hZ2UvbnVsbC8xNTMyMzk2MTM0Nzk1X3hvYXZ3d1Z5cGQuanBn?x-oss-process=image/format,png)

### 11.1. seed()

> np.random.seed更改随机数生成种子： 

```python
np.random.seed(1234)
```

>  numpy.random的数据生成函数使用了全局的随机种子。要避免全局状态，你可以使用numpy.random.RandomState，创建一个与其它隔离的随机数生成器

```python
rng = np.random.RandomState(1234)
print(rng.randn(10))
```

### 11.2. rand()均匀分布

均匀分布，生成(0,1)之间的数据。

```python
print(np.random.rand())
#生成一个数
#0.6548904423938792
 
print(np.random.rand(2,3))
#生成一组两行三列的数据
'''
[[0.07085309 0.52801719 0.2369186 ]
 [0.98840087 0.03015251 0.86048123]]
'''
print(np.random.rand(2,3)-0.5)
#生成一组(-0.5,0.5)之间的两行三列的数据
'''
[[-0.08586886  0.290878   -0.44895597]
 [-0.42226141  0.28728643 -0.33674102]]
'''
```

### 11.3. randint()

> randint给定上下限范围内的选取整数 

```python
import numpy as np
 
x=np.random.randint(5)             #一个参数，产生[0,n)内的整数
print(x)
#1
 
y=np.random.randint(0,10)          #两个参数，产生[a,b)内的整数
print(y)
#4
 
z1=np.random.randint(0,9,3)        #前两个参数是范围，第三个参数是元组类型的size
print(z1)
#[3 8 2]
 
z2=np.random.randint(0,10,(2,3))   
print(z2)
'''
[[1 7 7]
 [5 6 2]]
'''
```

Examples

```
>>> np.random.randint(2, size=10)
array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
>>> np.random.randint(1, size=10)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

```

Generate a 2 x 4 array of ints between 0 and 4, inclusive:

```
>>> np.random.randint(5, size=(2, 4))
array([[4, 0, 2, 1], # random
       [3, 2, 2, 0]])

```

Generate a 1 x 3 array with 3 different upper bounds

```
>>> np.random.randint(1, [3, 5, 10])
array([2, 2, 9]) # random

```

Generate a 1 by 3 array with 3 different lower bounds

```
>>> np.random.randint([1, 5, 7], 10)
array([9, 8, 7]) # random

```

Generate a 2 by 4 array using broadcasting with dtype of uint8

```
>>> np.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
array([[ 8,  6,  9,  7], # random
       [ 1, 16,  9, 12]], dtype=uint8)

```

### 11.4. randn()正态分布

> randn(d1,d2...)。不是元组类型

```python
x=np.random.randn(5)
print(x)
#[ 0.61697938  0.25645408  0.60145609  0.49726365  0.55797877]
y=np.random.randn(2,2)
print(y)
'''
[[-1.15277551  0.74185824]
 [-0.55820537  0.55722007]]
'''
```

## 12. 一个简单的随机漫步

## 13. 通过内置的random模块以纯Python的方式

> 一个通过内置的random模块以纯Python的方式实现1000步的随机漫步：从0开始，步长1和－1出现的概率相等。

```
import random
import matplotlib.pyplot as plt
%matplotlib inline
 
position = 0
#定义一个列表，现在里面只有一个元素0
walk = [position]
steps = 1000
#循环1000次
for i in range(steps):
     step = 1 if random.randint(0, 1) else -1
     #random.randint(0, 1)是产生随机的方向，或选择前者1的方向，或选择后者-1的方向
     #1和-1是设置步长
     position += step
     walk.append(position)
#输出前100次
plt.plot(walk[:100])
#这其实就是随机漫步中各步的累计和，可以用一个数组运算来实现。
```

![](https://img-blog.csdnimg.cn/20190324232154749.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbmRhbHBob240ODY5,size_16,color_FFFFFF,t_70)

## 14. 用np.random模块 

> 用np.random模块一次性就能随机产生1000个结果

```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
position = 0
#生成包含1000个元素的ndarray，元素要么是0，要么是1
arr=np.random.randint(0,2,size=1000)
#walk>0是判断条件，那么ndarray中的1值就变成1，0值就变成-1
step=np.where(arr>0,1,-1)
#walk就是前100个元素的累计和的ndarray，和纯python的walk列表一样
walk=step[:100].cumsum()
plt.plot(walk)
```

![](https://img-blog.csdnimg.cn/20190325170909742.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbmRhbHBob240ODY5,size_16,color_FFFFFF,t_70)

> 使用numpy模块，我们还可以统计一些数据

```
#最大值和最小值
print(walk.min())
print(walk.max())
#需要多久才能距离初始0点至少10步远（任一方向均可）
print((np.abs(walk) >= 10).argmax())
'''
-14
4
75
'''
#注意，这里使用argmax并不是很高效，因为它无论如何都会对数组进行完全扫描。
#在本例中，只要发现了一个True，那我们就知道它是个最大值了。
```

## 15. 一次模拟多个随机漫步

> 如果你希望模拟多个随机漫步过程（比如5000个），只需对上面的代码做一点点修改即可生成所有的随机漫步过程。只要给numpy.random的函数传入一个二元元组就可以产生一个二维数组，然后我们就可以一次性计算5000个随机漫步过程（一行一个）的累计和了：

```
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
#size=(5000,1000)意思是5000行，1000列，即5000个包含1000个元素的ndarray
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
 
print(walks)
'''
[[  1   2   3 ...,  38  39  38]
 [  1   0   1 ..., -32 -31 -32]
 [  1   0  -1 ...,  -6  -5  -4]
 ..., 
 [  1   0   1 ...,  -2  -3  -4]
 [  1   2   3 ...,  38  39  38]
 [  1   0   1 ...,  -4  -3  -2]]
'''
```

> 现在，我们来计算所有随机漫步过程的最大值和最小值：

```
print(walks.max())
print(walks.min())
'''
122
-130
'''
```

> 得到这些数据之后，我们来计算30或－30的最小穿越时间。这里稍微复杂些，因为不是5000个过程都到达了30。我们可以用any方法来对此进行检查：

```
hits30 = (np.abs(walks) >= 30).any(1)
print(hits30)
print(hits30.sum())
'''
[ True  True  True ..., False  True False]
3347
'''
```

> 然后我们利用这个布尔型数组选出那些穿越了30（绝对值）的随机漫步（行），并调用argmax在轴1上获取穿越时间：

```
#hits30意思是如果是False，那么肯定没有达到30，就跳过，这样比以此索引更高效
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
print(crossing_times.mean())
#506.527935465，平均穿越花费多长时间
```

> 用其他分布方式得到漫步数据。只需使用不同的随机数生成函数即可，如normal用于生成指定均值和标准差的正态分布数据：

```
steps = np.random.normal(loc=0, scale=0.25,size=(nwalks, nsteps))
```