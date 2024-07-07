- [1. dot product / scalar product / projection product](#1-dot-product--scalar-product--projection-product)
- [2. outer product](#2-outer-product)
- [3. Kronecker product](#3-kronecker-product)
- [4. 叉乘 Cross product](#4-叉乘-cross-product)
- [5. exterior product / wedge product](#5-exterior-product--wedge-product)
- [6. Hadamard product](#6-hadamard-product)


---

- **outer product**
- **dot product** (a special case of "**inner product**"), which takes a pair of coordinate vectors as input and produces a scalar
- **Kronecker product**, which takes a pair of matrices as input and produces a block matrix
- **Standard matrix multiplication**

## 1. dot product / scalar product / projection product

> 计算公式

$$\mathbf a \cdot \mathbf b = |\mathbf a| |\mathbf b| \cos \theta$$

如果在Cartesian Coordinates上，

$$\left\langle \mathbf {a} ,\mathbf {b} \right\rangle = \mathbf {a} ^ \top \mathbf {b} = \sum_{i=1}^{n}a_i b_i$$

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062037205.png)

> 性质

$$\begin{aligned}
&\vec{a}\cdot \vec{b}=\vec{b}\cdot\vec{a} & (\text{可交换}) \\
&\vec{a}\cdot(\vec{b}+\vec{c})=\vec{a}\cdot\vec{b}+\vec{a}\cdot\vec{c}  & (\text{分配律})\\
&(k\vec{a})\cdot\vec{b}=\vec{a}\cdot(k\vec{b})=k(\vec{a}\cdot\vec{b}) & (\text{结合律})
\end{aligned}$$

> **normalized vector** 的内积意义：如果都是单位向量，那么内积表示两个向量之间的夹角。 

$\cos\theta=\dfrac{a\cdot b}{\|a\|\|b\|}=a\cdot b$
    
- It is also used intensively to find out the angle between two vectors or compute the angle between a vector and the axis of a coordinate system (which is useful when the coordinates of a vector are converted to spherical coordinates.
- 单位向量间的内积结果，可以判断角度情况：
  - 点乘结果是0，向量互相垂直
  - 点乘结果是-1，向量方向相反
  - 点乘结果是-1，向量同一方向。

> 联系

The dot product is the trace of the outer product.

## 2. outer product
The outer product $\mathbf {u} \otimes_{\mathbf {outer}} \mathbf {v}$ is equivalent to a matrix multiplication $\mathbf {u} \mathbf {v} ^{\operatorname {T}}$.

$\mathbf {u}$ is represented as a $m\times 1$ column vector and $\mathbf {v}$ as a $n\times 1$ column vector.

For instance, if $\displaystyle m=4$ and $\displaystyle n=3$ then

$$\displaystyle \mathbf {u} \otimes \mathbf {v} 
=\mathbf {u} \mathbf {v} ^{\textsf {T}}
={\begin{bmatrix}u_{1}\\u_{2}\\u_{3}\\u_{4}\end{bmatrix}}{\begin{bmatrix}v_{1}&v_{2}&v_{3}\end{bmatrix}}={\begin{bmatrix}u_{1}v_{1}&u_{1}v_{2}&u_{1}v_{3}\\u_{2}v_{1}&u_{2}v_{2}&u_{2}v_{3}\\u_{3}v_{1}&u_{3}v_{2}&u_{3}v_{3}\\u_{4}v_{1}&u_{4}v_{2}&u_{4}v_{3}\end{bmatrix}}$$

- outer product is not commutative.
    $$\displaystyle {(\mathbf {u} \otimes \mathbf {v} )^{\textsf {T}}=(\mathbf {v} \otimes \mathbf {u} )}$$

```python
>>> a = np.array([0.0, 10.0, 20.0, 30.0])
>>> b = np.array([1.0, 2.0, 3.0])
>>> a[:, np.newaxis] * b
array([[ 0.,  0.,  0.],
       [10., 20., 30.],
       [20., 40., 60.],
       [30., 60., 90.]])
```

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062037206.png)  

## 3. Kronecker product

If $\displaystyle \mathbf {u} ={\begin{bmatrix}1&2&3\end{bmatrix}}^{\textsf {T}}$ and $\displaystyle \mathbf {v} ={\begin{bmatrix}4&5\end{bmatrix}}^{\textsf {T}}$ , we have:

$$\displaystyle {\begin{aligned}\mathbf {u} \otimes _{\text{Kron}}\mathbf {v} &={\begin{bmatrix}4\\5\\8\\10\\12\\15\end{bmatrix}} 
\\ \mathbf {u} \otimes _{\text{outer}}\mathbf {v} &={\begin{bmatrix}4&5\\8&10\\12&15\end{bmatrix}}\end{aligned}}$$

In the case of column vectors, the Kronecker product can be viewed as a form of vectorization (or flattening) of the outer product. In particular, for two column vectors $\mathbf {u}$ and $\mathbf {v}$ , we can write:

$$\displaystyle \mathbf {u} \otimes _{\text{Kron}}\mathbf {v} =\operatorname {vec} (\mathbf {v} \otimes _{\text{outer}}\mathbf {u} )$$



## 4. 叉乘 Cross product

https://en.wikipedia.org/wiki/Cross_product


$\mathbf a \times \mathbf b$ is a **vector** that is **perpendicular** to both $\mathbf a$ and $\mathbf b$, and thus normal to the plane containing them.

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062037207.png)

> 计算

$$\displaystyle \mathbf {a} \times \mathbf {b} =\left\|\mathbf {a} \right\|\left\|\mathbf {b} \right\|\sin(\theta )\ \mathbf {n}$$

Cartesian Coordinate: $\mathbf a = (x_a, y_a, z_a)^\top, \mathbf b = (x_b, y_b, z_b)^\top$: 

- $\mathbf {a\times b}=\begin{pmatrix}y_az_b-y_bz_a\\z_ax_b-x_az_b\\x_ay_b-y_ax_b\end{pmatrix}$

- $\displaystyle {\mathbf {a\times b} ={\begin{vmatrix}\mathbf {i} &\mathbf {j} &\mathbf {k} \\x_a&y_a&z_a\\x_b&y_b&z_b \end{vmatrix}}
=(y_az_b-z_ay_b)\mathbf {i} -(x_az_b-z_ax_b)\mathbf {j} +(x_ay_b-y_ax_b)\mathbf {k}}$

用对偶矩阵的矩阵乘法实现叉乘：

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062037208.png)

> 向量方向

Direction determined by right-hand rule: 
- 手掌：右手，四指由 a（第一个向量） 转到 b（第二个向量），大拇指朝向就是aXb的方向。注意是握拳（从石头剪刀布的布到点赞手势），而不是摊掌。
- 三指：右手，大拇指a，食指b，中指的方向就是axb。（是大食中、食中大、中大食的升序，而不是中食大等的降序）

故而，交换位置，方向相反。

$$\begin{aligned}
&\mathbf a \times \mathbf b=-\mathbf b \times \mathbf a & (\text{不可交换}) \\
\end{aligned}$$

> 性质


$$\begin{aligned}
&\mathbf a \times \mathbf b=-\mathbf b \times \mathbf a & (\text{不可交换}) \\
&\vec{\mathbf a}\times\vec{\mathbf a}=\vec{\mathbf 0} \\
&\vec{\mathbf a}\times(\vec{\mathbf b}+\vec{\mathbf c})=\vec{\mathbf a}\times\vec{\mathbf b}+\vec{\mathbf a}\times\vec{\mathbf c} & (\text{分配律}) \\
&\vec{\mathbf a}\times(k\vec{\mathbf b})=（k\vec{\mathbf a})\times\vec{\mathbf b}=k(\vec{\mathbf a}\times\vec{\mathbf b})& (\text{结合律}) \\
\end{aligned}$$

> 乘积的大小等于具有边向量的平行四边形的面积. 
- If the vectors a and b are parallel (that is, the angle θ between them is either 0° or 180°), by the above formula, the cross product of a and b is the zero vector 0.
- 还在这条线上，保持d的长度不变，滑动这条线段，乘积不变。

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062037209.png)
    

> 意义：判断左右、内外

- 左右: 如何判断向量a和b的左右关系（逆时针，顺时针）
  
  向量a和b在xoy平面上

  ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062037210.png)

  - $\vec a \times \vec b$ 是正的（向外），则说明b在a的左侧
  - $\vec b \times \vec a$ 是负的（向内），则说明a在b的右侧

- 内外：如何判断点p在三角形的内部

  ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062037211.png)

  $\vec {AB} \times \vec{AP}$、$\vec {BC} \times \vec{BP}$、$\vec {CA} \times \vec{CP}$。P在AB、BC、AC左侧，即在三角形内。

  
## 5. exterior product / wedge product

$$\displaystyle {u\wedge v}$$

$$\displaystyle {u\wedge v=-(v\wedge u)}$$


In connection with the cross product, the exterior product of vectors can be used in arbitrary dimensions (with a bivector or 2-form result) and is independent of the orientation of the space.

## 6. Hadamard product

element-wise, 按元素相乘