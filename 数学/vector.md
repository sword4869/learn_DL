- [1. direction of vector from points](#1-direction-of-vector-from-points)
- [2. 模](#2-模)
- [3. normalize\_vec](#3-normalize_vec)
- [4. Projection](#4-projection)
- [5. 施密特正交化](#5-施密特正交化)

---
## 1. direction of vector from points

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062038163.png)

## 2. 模

在欧式空间上，向量的模即向量的二范数。

$|a|=\sqrt{\langle a,a\rangle} == \displaystyle {\left\| X \right\|_2 } =\sqrt{\sum_{i=1}^{n} x_i^2 }$

## 3. normalize_vec

$\hat a=a/\|a\|$

```python
import numpy as np

def normalize_vec(x:np.ndarry):
    '''
    向量的单位化: 向量除以自己的二范数，得到和这个向量方向相同的单位向量(向量的范数为1)。
    '''
    return x / np.linalg.norm(x)


def normalize_features(x):
    '''
    归一化，所有的feature位于[0.0, 1.0]之间.
    所谓uint8[0, 255]图像到float[0.0, 1.0], img = img / 255。即 (x - 0) / (255 - 0)，这里0和255必须是指定的，而不是 x.max()和x.min()。
    '''
    # (x.max() - x/min()) 是一个scalar
    # (x - x.min()) 是 ndarry 对 scalar的减法，还是一个ndarry
    # 然后除法后还是 ndarry
    return (x - x.min()) / (x.max() - x/min())


def normalize_double(x):
    '''
    每个值均在 [-1.0, 1.0] 之间
    '''
    # 由[0,1]，再[0,2]，再[-1,1]
    n1 = normalize_features(x)
    n2 = n1 * 2 - 1
    return n2
```

## 4. Projection

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062038164.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062038165.png) 

将向量分解为互相垂直的向量

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062038166.png)

投影到正交坐标系：

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062038167.png)

## 5. 施密特正交化
> 分步

1. 正交向量组
  
    $$\begin{aligned}
    \beta_1 &=\alpha_1 \\
    \beta_{2}&=\alpha_{2} -\frac{\langle\alpha_2,\beta_1\rangle}{\langle\beta_1,\beta_1\rangle}\beta_1 \\
    \\
    \beta_{m}&=\alpha_{m}-{\frac{\langle\alpha_{m},\beta_{1}\rangle}{\langle\beta_{1},\beta_{1}\rangle}}\beta_{1}-{\frac{\langle\alpha_{m},\beta_{2}\rangle}{\langle\beta_{1},\beta_{2}\rangle}}\beta_{2}-\cdots-{\frac{\langle\alpha_{m},\beta_{m-1}\rangle}{\langle\beta_{m-1},\beta_{m-1}\rangle}}\beta_{m-1} 
    \end{aligned}
    $$

2. 再单位化就是满足要求的标准正交向量组

    $$e_i=\frac{\beta_i}{\|\beta_i\|}$$

> 直接二合一得到标准正交向量组：

$a'=a/\|a\|,  b'=b-(b\cdot a')a'$ 

其实就是 

$b'= b-\dfrac{\langle b, a'\rangle}{\sqrt{\langle a,a\rangle}}a = b-\dfrac{\langle b, a\rangle}{\langle a,a \rangle}a$