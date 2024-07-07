- [1. model transformation 模型变换](#1-model-transformation-模型变换)
  - [存疑](#存疑)
- [2. camera/view transformation 视角变换(w2c)](#2-cameraview-transformation-视角变换w2c)
  - [存疑](#存疑-1)
  - [2.1. 在不同坐标系间坐标的转换](#21-在不同坐标系间坐标的转换)
    - [2.1.1. frame-to-canonical matrix (正向)](#211-frame-to-canonical-matrix-正向)
    - [2.1.2. canonical-to-frame matrix (反向)](#212-canonical-to-frame-matrix-反向)
  - [2.2. 右手坐标系 right-handed coordinates](#22-右手坐标系-right-handed-coordinates)
  - [2.3. 相机](#23-相机)
  - [2.4. 各种右手的相机坐标系的转换](#24-各种右手的相机坐标系的转换)
    - [2.4.1. 在获取c2w时](#241-在获取c2w时)
    - [2.4.2. 已经获取c2w后](#242-已经获取c2w后)
  - [2.5. 其他](#25-其他)
- [3. projection transformation](#3-projection-transformation)
  - [3.1. orthographic projection](#31-orthographic-projection)
    - [3.1.1. view volume to canonical view volume](#311-view-volume-to-canonical-view-volume)
    - [3.1.2. viewporrt transformation](#312-viewporrt-transformation)
  - [3.2. perspective projection](#32-perspective-projection)
    - [3.2.1. frustum to canonical view volume](#321-frustum-to-canonical-view-volume)

---

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062023181.png)

图像的成像过程经历了世界坐标系—>相机坐标系—>图像坐标系—>像素坐标系这四个坐标系的转换，如下图所示：


![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062024743.png)
- 像素坐标系 pixels coordinate：以图像平面左上角为原点的坐标系 ，X 轴和Y 轴分别平行于图像坐标系的 X 轴和Y 轴，用 $(u,v)$ 表示其坐标值。像素坐标系就是以像素为单位的图像坐标系。

- 图像坐标系 image coordinate：以光心在图像平面投影为原点的坐标系 ，X轴和Y 轴分别平行于图像平面的两条垂直边，用 $(x, y)$ 表示其坐标值。图像坐标系是用物理单位表示像素在图像中的位置。

- 相机坐标系 camera coordinate：以相机光心为原点的坐标系，X 轴和Y 轴分别平行于图像坐标系的 X 轴和Y 轴，相机的光轴为Z 轴，用 $(x_{c}, y_{c},z_{c})$ 表示其坐标值。

- 世界坐标系 world coordinate：是三维世界的绝对坐标系，我们需要用它来描述三维环境中的任何物体的位置，用 $(x_{w}, y_{w},z_{w})$ 表示其坐标值。


## 1. model transformation 模型变换

world coordinate

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062024109.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062024956.png)

如何放置模型坐标到世界坐标中。

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062024700.png)


SRT: scale, rotation, translation. 

顺序也是先 scale, 后rotation, 最后translation（S和R是线性变换，故可交换顺序，而T不行，必须放在最后）。

- 第一种: 
    $$\begin{aligned}
    P_{c}&=RSP_{w}+t
    \end{aligned}$$

- 第二种
  
    $$\begin{aligned}
    \begin{bmatrix} P_{c} \\ 1\end{bmatrix}&=\begin{bmatrix} R & t \\ 0^\top & 1 \end{bmatrix} \begin{bmatrix} S & 0 \\ 0^\top & 1 \end{bmatrix} \begin{bmatrix} P_{w} \\ 1\end{bmatrix}    
    \end{aligned}$$
### 存疑
从世界坐标系到世界坐标系。所以操作都是以物体的角度看。
- 模型的中心一开始就在世界坐标系原点。
- 不是旋转坐标系，而是旋转人头；
- 不是翻转y轴，而是翻转人头；
- 不是平移坐标系，而是平移人头
- 先旋转后要看正脸还是侧脸后，我们把模型沿着z轴平移（look at)，因为中点在人头内部，看不到uv。

```python
def pose_spherical(radius, theta, phi):
    '''
    theta: 方位角
    phi: 极角
    return: [4, 4]
    '''
    pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    pose = rot_theta(theta/180.*np.pi) @ pose   # y
    pose = rot_phi(phi/180.*np.pi) @ pose       # x
    pose = trans_t(radius) @ pose
    pose = pose @ np.diag([1, -1, 1, 1])        # 这b是倒吊人
    return pose
```

## 2. camera/view transformation 视角变换(w2c)

### 存疑
**将相机坐标系转到与世界坐标系重合：先旋转轴来轴向一致，再将相机平移到世界原点; M=RT， 先平移再旋转**。

将相机和物体一起变换。所以相机坐标系下的物体坐标，变换矩阵乘物体的世界坐标。(变换点，就是变换整个坐标系)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062024160.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062024498.png)


乘了这个后，再乘别的变换矩阵，就是从原点移动相机矩阵的位置。

所以外参中的xyz是负的相机原点在世界坐标系的位置。

---

### 2.1. 在不同坐标系间坐标的转换

仿射变换：从uv到xy，就要知道原坐标系 $uve$ 在目标坐标系xy的表示，然后得到 $tR$

逆仿射变换：从xy到uv，知道目标坐标系 $uve$ 在原坐标系xy的表示，然后得到 $R^{-1}t^{-1}$


#### 2.1.1. frame-to-canonical matrix (正向)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062024531.png)

$\mathbf{p}=(x_p,y_p)\equiv\mathbf{o}+x_p\mathbf{x}+y_p\mathbf{y}$

$\mathbf{p}=(u_{p},v_{p})\equiv\mathbf{e}+u_{p}\mathbf{u}+v_{p}\mathbf{v}$

已知点p在uv坐标系的表示 $(u_{p},v_{p})$ 和 uv坐标轴在xy坐标系的单位向量 $\mathbf{u}\mathbf{v}$ 和 uv坐标系原点在xy坐标系的位置 $\mathbf{e}$，那么就可以求 $(x_p,y_p)$: 

$\begin{aligned}
\begin{bmatrix}x_p \\ y_p \\ 1\end{bmatrix} &= u_p\begin{bmatrix}x_u \\ y_u \\0\end{bmatrix}v_p\begin{bmatrix}x_v \\ y_v \\0\end{bmatrix} + \begin{bmatrix}x_e \\ y_e \\1\end{bmatrix} \\
&= \begin{bmatrix}x_u&x_v&x_e\\y_u&y_v&y_e\\0&0&1\end{bmatrix}\begin{bmatrix}u_p\\v_p\\1\end{bmatrix} &(\text{写成矩阵形式})\\
&= \begin{bmatrix}1&0&x_e\\0&1&y_e\\0&0&1\end{bmatrix}\begin{bmatrix}x_u&x_v&0\\y_u&y_v&0\\0&0&1\end{bmatrix}\begin{bmatrix}u_p\\v_p\\1\end{bmatrix} &\text{(拆分仿射变换)}\\
&= TRP_{uv}
\end{aligned}$

由此，视角变换矩阵的列向量意义：$\mathbf{p}_{xy}=\begin{bmatrix}\mathbf{u}&\mathbf{v}&\mathbf{e}\\0&0&1\end{bmatrix}\mathbf{p}_{uv}$
- R：uv坐标系有两个坐标分量，就有两个列向量对应。这两个列向量，分别是uv坐标轴在xy坐标系的单位向量表示。
- T: 最后一列，uv坐标系的原点在xy坐标系的位置 $\mathbf{e}$。

#### 2.1.2. canonical-to-frame matrix (反向)

$\mathbf{p}_{uv}=\begin{bmatrix}\mathbf{u}&\mathbf{v}&\mathbf{e}\\0&0&1\end{bmatrix}^{-1}\mathbf{p}_{xy}$

$\begin{bmatrix}\mathbf{u}&\mathbf{v}&\mathbf{e}\\0&0&1\end{bmatrix}^{-1} = (TR)^{-1} = R^{-1}T^{-1} = \begin{bmatrix}x_u&y_u&0\\x_v&y_v&0\\0&0&1\end{bmatrix}\begin{bmatrix}1&0&-x_e\\0&1&-y_e\\0&0&1\end{bmatrix} = \begin{bmatrix} x_u & y_u & -x_u x_e -y_u y_e \\ x_v & y_v & -x_v x_e y_v y_e \\ 0 & 0 & 1\end{bmatrix}$

> 困惑: 那这反向的矩阵的过程代表什么意义？

反向的过程仅仅是矩阵求逆，所以过程没有像正向一样有意义，只是逆变换而已。

$\begin{bmatrix}x_u&y_u&0\\x_v&y_v&0\\0&0&1\end{bmatrix}\begin{bmatrix}1&0&-x_e\\0&1&-y_e\\0&0&1\end{bmatrix}$ 可以解释为先平移后旋转。

此时平移看似是平移负量，但是此时是在 uv坐标系下，你拿xy坐标系的位置表示干什么，要拿也是拿uv坐标系下看xy坐标轴的原点的表示啊。

只是结果有意义， $\begin{bmatrix} x_u & y_u & -x_u x_e -y_u y_e \\ x_v & y_v & -x_v x_e y_v y_e \\ 0 & 0 & 1\end{bmatrix}$ 可以解读为xy坐标轴在uv坐标系的向量表示 和 原点位置 $(-x_u x_e -y_u y_e , -x_v x_e y_v y_e)$.

> 那么谁是正向，谁是反向？

正向反向，只看已知条件是由谁表示的。

比如，我们可以讲 xy→uv是正向，那么就需要知道：点p在xy坐标系的表示 $(x_{p},y_{p})$ 和 xy坐标轴在uv坐标系的单位向量 $\mathbf{x_{uv}}\mathbf{y_{uv}}$ 和 xy坐标系原点在uv坐标系的位置 $\mathbf{o_{uv}}$，那么就可以求 $(u_p,v_p)$

$\mathbf{p}_{uv}=\begin{bmatrix}\mathbf{x}_{uv}&\mathbf{y}_{uv}&\mathbf{o}_{uv}\\0&0&1\end{bmatrix}\mathbf{p}_{xy}$

### 2.2. 右手坐标系 right-handed coordinates

In 2D, right-handed means y is counterclockwise from x.


> 将左右手性和right-up-forward联系在一起，而不是xyz

The only thing that defines the handedness of the coordinate system is the orientation of the left (or right) vector relative to the up and forward vectors, regardless of what these axes represent.

- 右手：选一个判断就行————$\text{up} \times \text{forward} = + \text{right}$
  
    右手坐标系的6个性质：
    $$\begin{aligned}
    {right}\times{up}&=+{forward} \\
    {up}\times{forward}&=+{right} \\
    {forward}\times{right}&=+{up} \\
    {up}\times{right}&=-{forward} \\
    {forward}\times{up}&=-{right} \\
    {right}\times{forward}&=-{up}
    \end{aligned}$$
- 左手：$\text{up} \times \text{forward} = - \text{right}, 即 +\text{left}$
  
    左手坐标系的6个性质：
    $$\begin{aligned}
    {right}\times{up}&=-{forward} \\
    {up}\times{forward}&=-{right} \\
    {forward}\times{right}&=-{up} \\
    {up}\times{right}&=+{forward} \\
    {forward}\times{up}&=+{right} \\
    {right}\times{forward}&=+{up}
    \end{aligned}$$


手掌：用右手的**4个指头从a转向b**（合拳，而不是松拳），大拇指朝向就是aXb的方向。

三指：右手，大拇指a，食指b，中指的方向就是axb。（是大食中、食中大、中大食的升序，而不是中食大等的降序）


![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062026967.png)

![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062026968.png)

图中b还是右手性，是认为up是z轴，按照right-up-forward来判断它还是右手性。


> 目前，默认采用图a的方式，+X is right, +Y is up, and +Z is forward

判断左右手坐标系：
- 右手坐标系：$\vec{x}\times\vec{y}=+\vec{z}$
- 左手坐标系：$\vec{x}\times\vec{y}=-\vec{z}$

右手坐标系的6个性质：
$$\begin{aligned}
&\vec{x}\times\vec{y}=+\vec{z} \\
&\vec{y}\times\vec{z}=+\vec{x} \\
&\vec{z}\times\vec{x}=+\vec{y} &\text{特别记住}\\
&\vec{y}\times\vec{x}=-\vec{z}\\
&\vec{z}\times\vec{y}=-\vec{x} \\
&\vec{x}\times\vec{z}=-\vec{y}  &\text{特别记住}
\end{aligned}$$

> 旋转不变性

一个坐标系是左(右)手坐标系, 如果我们把手转90°，这依旧是一个左(右)手坐标系。

The only thing that defines the handedness of the coordinate system is the orientation of the left (or right) vector relative to the up and forward vectors, regardless of what these axes represent.

> 约定俗成的配置：
- x points to the right
- y is up
- z is backwards (look at -z) (coming out of the screen).
  
    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062028778.png)

### 2.3. 相机

> c2w: camera coordinate to world coordinate

camera's pose matrix in world coordinate.

The camera's extrinsic matrix describes 
- $\mathbf{R}$: 3 columns are the +X, +Y, and +Z defining the camera orientation in world coordinate. 
  
    为什么代表旋转方向？因为相机不同的旋转方向，对应不同的坐标轴向量。
- $\mathbf{t}$: 相机坐标系的原点，也是 the camera's location in the world

> column-major w2c↔c2w，两矩阵互逆。

$$
\begin{aligned}
\left[\begin{array}{c|c}\mathbf{R_c}&\mathbf{C}\\\hline \mathbf{0}\top&1\end{array}\right]
& = \left[\begin{array}{c|c}\mathbf{R}&\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]^{-1}  \\
&=\left[\left[\begin{array}{c|c}\mathbf{I}&\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]\left[\begin{array}{c|c}\mathbf{R}&0\\\hline\mathbf{0}&1\end{array}\right]\right]^{-1}& (\text{decomposing rigid transform})  \\
&=\left[\begin{array}{c|c}\mathbf{R}&0\\\hline\mathbf{0}\top&1\end{array}\right]^{-1}\left[\begin{array}{c|c}\mathbf{I}&\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]^{-1}& (\text{distributing the inverse})  \\
&=\left[\begin{array}{c|c}\mathbf{R}^\top &0\\\hline\mathbf{0}\top&1\end{array}\right]\left[\begin{array}{c|c}\mathbf{I}&-\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]& \text{(applying the inverse)}  \\
&=\left[\begin{array}{c|c}\mathbf{R}^\top&-\mathbf{R}^\top\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]& (\text{matrix multiplication}) 
\end{aligned}
$$


即 $T_{w2c} = [\mathbf{R}, \mathbf{t}], 则T_{c2w} = T_{w2c}^{-1} = [\mathbf{R}^\top, -\mathbf{R}^\top\mathbf{t}]$

```python
c2w = np.linalg.inv(w2c)
```

### 2.4. 各种右手的相机坐标系的转换

![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062028328.png)

![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062028296.png)

注意：各种右手的相机坐标系的转换，在c2w和w2c上表现不同。

#### 2.4.1. 在获取c2w时

w2c 和 c2w，谁设正向都可以，正向的才有意义。但是为什么我们设c2w为正向。

因为我们很容易表达相机坐标轴在世界坐标的位置（即下面的方法），而不容易表达世界坐标轴在相机坐标系的位置。

有相机坐标的原点 $e$, 相机 look at 方向的向量 $\vec v$, 相机的向上向量 $\vec{up}$. 这三者均是在世界坐标系的表示。


![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062031756.png)

```python
# 这里以 RDF 举例
e = np.array([0, 0, 500], dtype=np.float32)
v = np.array([0, 0, -1], dtype=np.float32)  # D
# v = np.array([0, 0, 1], dtype=np.float32)  # U
up = np.array([0, -1, 0], dtype=np.float32)

w = normalize(v)                    # F
# w = =normalize(v)                    # B
r = normalize(np.cross(up, v))
u = normalize(np.cross(w, r))

c2w = np.zeros((4, 4), dtype=np.float32)
c2w[:3, 0] = r  # x
c2w[:3, 1] = u  # y
c2w[:3, 2] = w  # z
c2w[:3, 3] = e
c2w[3, 3] = 1
print(c2w)
w2c = np.linalg.inv(c2w)
print(w2c)
```


#### 2.4.2. 已经获取c2w后

- c2w->c2w: 只动R，变的是轴（整个列向量）

```python
# R: [3, 3], c2w: [3, 4] or [4, 4]

# RDF to RUB: x'=x, y'=-y, z'=-z
R[:, 1:3] *= -1
c2w[:3, 1:3] *= -1

R = np.concatenate([R[:, 0:1], -R[:, 1:2], -R[:, 2:3]], 1)
c2w = np.concatenate([c2w[:, 0:1], -c2w[:, 1:2], -c2w[:, 2:3], c2w[:, 3:4]], 1)
R = R @ np.diag([1, -1, -1])
c2w = c2w @ np.diag([1, -1, -1, 1])
```

涉及非对称
```python
# RDF to DRB: x'=y, y'=x, z'=-z
R = np.concatenate([R[:, 1:2], R[:, 0:1], -R[:, 2:3]], 1)
c2w = np.concatenate([c2w[:, 1:2], c2w[:, 0:1], -c2w[:, 2:3], c2w[:, 3:4]], 1)
R = R @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
c2w = c2w @ np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
```

> w2c->w2c: 变的是各个列向量的xyz坐标

因为 $T_{w2c} = T_{c2w}^{-1} = [\mathbf{R}^\top, -\mathbf{R}^\top\mathbf{t}]$.  所以，c2w的R列变化，变成了w2c的行变化，而且T也跟着变。

```python
# RDF to RUB: x'=x, y'=-y, z'=-z
w2c[1:3, :] = -w2c[1:3, :]
w2c = np.diag([1, -1, -1, 1]) @ w2c
```
涉及非对称
```python
# RDF to DRB: x'=y, y'=x, z'=-z
w2c = np.concatenate([w2c[1:2], w2c[0:1], -w2c[2:3], w2c[3:4]], 0)
w2c = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ w2c
```
### 2.5. 其他

$$\begin{aligned}
&R_{w\to 2} \left[ R_{1\to w}\begin{bmatrix}x \\ y \\ z \end{bmatrix} + t_{1\to w} \right] + t_{w\to 2} \\
&= R_{w\to 2} R_{1\to w} \begin{bmatrix}x \\ y \\ z \end{bmatrix} + R_{w\to 2} t_{1\to w} + t_{w\to 2} \tag{1}\\
&= R_{1\to 2}\begin{bmatrix}x \\ y \\ z \end{bmatrix} + t_{1\to 2}&\text{(得到视角之间的变换)} 
\end{aligned}$$

代码：
```python
R = R_2 @ R_1     # [3, 3]
t = R_2 @ t_1 + t_2   # [3]
extrin = np.hstack((R, t[None].T))  # [3, 4]，如何将R和T拼接
```
## 3. projection transformation

perspective projection 透射投影 和 orthographic projection 正交投影 的区别：无有近大远小。

### 3.1. orthographic projection

#### 3.1.1. view volume to canonical view volume

将三维空间投影至标准二维平面($[-1,1]^2$)之上 

orthographic projection is affine transformation (最后一行是 $\begin{bmatrix} 0 & 0 & 0 & 1\end{bmatrix}$)

> 不常用的做法

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062031757.png)

1. 直接把 camera coordinate 下的 z 坐标扔掉
2. Translate and scale the resulting rectangle to $[-1, 1]^2$ .  

> 常用的做法

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062031758.png)

1. find a **cuboid** [l, r] x [b, t] x [f, n]. 按照此相机坐标系，look at -z, near > far.
2. map to the “canonical (正则、规范、标准)” cube $[-1, 1]^3$ . 先平移，再缩放。

$M_{ortho}=\mathbf{ST}=\begin{bmatrix}\frac{2}{r-l}&0&0&0\\0&\frac{2}{t-b}&0&0\\0&0&\frac{2}{n-f}&0\\0&0&0&1\end{bmatrix}\begin{bmatrix}1&0&0&-\frac{r+l}2\\0&1&0&-\frac{t+b}2\\0&0&1&-\frac{n+f}2\\0&0&0&1\end{bmatrix} = \begin{bmatrix}\frac{2}{r-l}&0&0&-\frac{r+l}{r-l}\\0&\frac{2}{t-b}&0&-\frac{t+b}{t-b}\\0&0&\frac{2}{n-f}&-\frac{n+f}{n-f}\\0&0&0&1\end{bmatrix}$

PS: 这里的z并没有丢掉，为了之后的遮挡关系检测
#### 3.1.2. viewporrt transformation

将处于标准平面映射到屏幕分辨率范围之内，即$[-1,1]^2 \rightarrow [0,width]*[0,height]$, 其中width和height指屏幕分辨率大小.

$M_{viewport}=\begin{pmatrix}\frac{width}{2}&0&0&\frac{width}{2}\\0&\frac{height}{2}&0&\frac{height}{2}\\0&0&1&0\\0&0&0&1\end{pmatrix}$

完整的正交投影即，$M = M_{viewport}M_{ortho}$



### 3.2. perspective projection

perspective projection (最后一行是 $\begin{bmatrix} 0 & 0 & 1 &0\end{bmatrix}$ ) is **not** affine transformation ($\begin{bmatrix} 0 & 0 & 0 & 1\end{bmatrix}$)

#### 3.2.1. frustum to canonical view volume

这个的投影考虑视锥的 left, right, top, bottom, near, far

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062031760.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062031761.png)

将透射投影分解为 $M = M_{ortho}M_{persp-ortho}$：
- First “squish” the frustum into a cuboid (n -> n, f -> f) ($M_{persp-ortho}$). 
    把f面挤压成和n面一样大，是为了确定f面中的物体投影到n面上的大小，再进行正交投影即可。

    所以近大远小可以解释为，近处的被挤压的程度小，远处的被挤压的程度大。
- Do orthographic projection ($M_{ortho}$)

那么如何得到 $M_{persp-ortho} = \begin{bmatrix}?&?&?&?\\?&?&?&?\\?&?&?&?\\?&?&?&?\end{bmatrix}$

$\begin{bmatrix}x^\prime \\ y^\prime \\ z^\prime \\ 1 \end{bmatrix} = M_{persp-ortho}\begin{bmatrix}x \\ y \\ z \\ 1 \end{bmatrix}$

> 第一个观察：我们发现 x/y 被挤压后的坐标 x'/y', 刚好可以根据在近平面上的相似三角形计算。PS：z'不是不变，而且是非相似三角形的变动。

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062031762.png)


$\begin{bmatrix}x^\prime \\ y^\prime \\ z^\prime \\ 1 \end{bmatrix} = \begin{bmatrix}nx/z \\ ny/z \\ ?  \\ 1 \end{bmatrix} == \begin{bmatrix}nx \\ ny \\ ?  \\ z \end{bmatrix}$， 这里和之后齐次坐标都是乘以原来点的z位置，不是乱乘的。

从而有 $M_{persp-ortho}\begin{bmatrix}x \\ y \\ z \\ 1 \end{bmatrix} = \begin{bmatrix}nx \\ ny \\ ?  \\ z \end{bmatrix}$

推测出 $M_{persp-ortho} = \begin{bmatrix}n&0&0&0\\0&n&0&0\\?&?&?&?\\0&0&1&0\end{bmatrix}$

> 第二个观察：近远平面上的点的特点
- Any point on the near plane will not change. x'=x, y'=y, z'=z=n
  
  $M_{persp-ortho}\begin{bmatrix}x \\ y \\ n \\ 1 \end{bmatrix} = \begin{bmatrix}x \\ y \\ n \\ 1 \end{bmatrix} == \begin{bmatrix}nx \\ ny \\ n^2  \\ n \end{bmatrix}$, 即$M_{persp-ortho}\begin{bmatrix}x \\ y \\ n \\ 1 \end{bmatrix} = \begin{bmatrix}nx \\ ny \\ n^2  \\ n \end{bmatrix}$

  则有第三行 $\begin{bmatrix} ? & ? & ? & ? \end{bmatrix}\begin{bmatrix}x \\ y \\ n \\ 1 \end{bmatrix} = n^2$

  推测出前两个系数与x和y无关，是0. 则只剩两个未知系数，$\begin{bmatrix} 0 & 0 & A & B\end{bmatrix}$

  且(1)，$An+B=n^2$

- Any point’s z on the far plane will not change. z'=z=f



  取远平面的中心点

  $M_{persp-ortho}\begin{bmatrix}0 \\ 0 \\ f \\ 1 \end{bmatrix} = \begin{bmatrix}0 \\ 0 \\ f \\ 1 \end{bmatrix} == \begin{bmatrix}0\\0 \\ f^2  \\ f \end{bmatrix}$, 即$M_{persp-ortho}\begin{bmatrix}0 \\ 0 \\ f \\ 1 \end{bmatrix} = \begin{bmatrix}0\\0 \\ f^2  \\ f \end{bmatrix}$

  则有第三行 $\begin{bmatrix} 0 & 0 & A & B \end{bmatrix}\begin{bmatrix}0 \\ 0 \\ f \\ 1 \end{bmatrix} = f^2$


  且(2)，$Af+B=f^2$

联立(1)(2)，有 $A=n+f, B=-nf$

综上，$M_{persp-ortho} = \begin{bmatrix}n&0&0&0\\0&n&0&0\\0 &0 & n+f & -nf\\0&0&1&0\end{bmatrix}$

$M = M_{ortho}M_{persp-ortho} = \begin{bmatrix}\frac{2}{r-l}&0&0&-\frac{r+l}{r-l}\\0&\frac{2}{t-b}&0&-\frac{t+b}{t-b}\\0&0&\frac{2}{n-f}&-\frac{n+f}{n-f}\\0&0&0&1\end{bmatrix} 
\begin{bmatrix}n&0&0&0\\0&n&0&0\\0 &0 & n+f & -nf\\0&0&1&0\end{bmatrix}
= \begin{bmatrix}\frac{2n}{r-l}&0&-\frac{r+l}{r-l}&0\\0&\frac{2n}{t-b}&-\frac{t+b}{t-b}&0\\0&0&\frac{n+f}{n-f}&-\frac{2nf}{n-f}\\0&0&1&0\end{bmatrix}$

没加 viewport transformation.

#### 3.2.2. pinhole 的 K矩阵

##### 3.2.2.1. 相机坐标系->图像坐标系

这里的相似三角形就不是任意值的近平面了，而是 image plane (the distance between the pinhole and image plane is **focal length**.)


![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062027590.png)

图像坐标系(对应平面叫做image plane)的x和y轴方向和相机坐标系的保持一致。

![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062027591.png)

从3D的相机坐标系下的欧式点 $(X_{c}, Y_{c}, Z_{c})$ 到2D的图像坐标系下的欧式点 $(x,y)$  

$$\begin{aligned}
\dfrac{f}{Z_{c}} &= \dfrac{x}{X_c} = \dfrac{y}{Y_c}
\\ x&=f\dfrac{X_{c}}{Z_{c}}
\\ y&=f\dfrac{Y_{c}}{Z_{c}}\end{aligned}$$

$\begin{bmatrix} f_x & 0 & 0 & 0\\ 0 & f_y & 0 & 0\\ 0 & 0 & 1 & 0\end{bmatrix}  \begin{bmatrix} X_{c} \\  Y_{c} \\ Z_{c} \\ 1 \end{bmatrix}
=\begin{bmatrix} f_xX_c \\ f_yY_c \\ Z_c \end{bmatrix} 
=Z_c\begin{bmatrix} f_x\dfrac{X_{c}}{Z_{c}} \\ f_y\dfrac{Y_{c}}{Z_{c}} \\ 1 \end{bmatrix}
=Z_c\begin{bmatrix}x \\y \\1 \end{bmatrix}$

$\begin{bmatrix} f_x & 0 & 0\\ 0 & f_y & 0\\ 0 & 0 & 1\end{bmatrix}  \begin{bmatrix} X_{c} \\  Y_{c} \\ Z_{c}\end{bmatrix}
=\begin{bmatrix} f_xX_c \\ f_yY_c \\ Z_c \end{bmatrix} 
=Z_c\begin{bmatrix} f_x\dfrac{X_{c}}{Z_{c}} \\ f_y\dfrac{Y_{c}}{Z_{c}} \\ 1 \end{bmatrix}
=Z_c\begin{bmatrix}x \\y \\1 \end{bmatrix}$


PS：倒像问题


![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062025018.png)

P点的x是负坐标，P'点的x是正坐标。

$$\dfrac{f}{Z_{c}} = -\dfrac{x}{X_c} = -\dfrac{y}{Y_C}$$

其中负号表示成的像是倒立的。为了简化模型，我们可以把成像平面对称到相机前方，和三维空间点一起放在摄像机坐标系的同一侧，这样做可以把公式中的负号去掉，使式子更加简洁。


![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062025565.png)

##### 3.2.2.2. 图像坐标系->像素坐标系

![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062025778.png)

像素坐标系：以左上角点为原点，u轴向右与x轴平行，v轴向右与y轴平行。像素坐标系和图像坐标系之间，相差了一个缩放 $\alpha, \beta$和原点的平移 $c_x, c_y$。

$$\begin{aligned}
u&=\alpha x + c_x
\\ v&=\beta y + c_y
\end{aligned}$$

$
Z_c \begin{bmatrix}\alpha & 0 & c_x \\0 & \beta & c_y \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix}x \\y \\1 \end{bmatrix} = 
Z_c\begin{bmatrix} u \\v \\ 1\end{bmatrix}
$
##### 3.2.2.3. 相机内参

The intrinsic matrix transforms 3D camera cooordinates to 2D homogeneous image coordinates.

$K = \begin{bmatrix} \alpha f_x & s & c_x\\ 0 & \beta f_y & c_y\\ 0 & 0 & 1\end{bmatrix}$


![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062027592.png)

平移操作放在最后（即最左）。

在内参矩阵中还有个参数 $s$（通常也可当成0），用来建模像素是平行四边形而不是矩形，与像素坐标系的u，v轴之间的夹角$\theta$的正切值$tan(\theta)$成反比，因此当 $s = 0$时，表示像素为矩形。

$Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} = KP_c$

1. 3行3列

$$\begin{aligned}
Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix}
&=\begin{bmatrix} \alpha & 0 & c_x \\0 & \beta & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} f_x & 0 & 0\\ 0 & f_y & 0\\ 0 & 0 & 1\end{bmatrix} \begin{bmatrix} X_{c} \\  Y_{c} \\ Z_{c} \end{bmatrix}\\
&=\begin{bmatrix} \alpha f_x & 0 & c_x\\ 0 & \beta f_y & c_y\\ 0 & 0 & 1\end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c\end{bmatrix}
\end{aligned}
$$
2. 3行4列

$$\begin{aligned}
Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} 
&= \begin{bmatrix} \alpha & 0 & c_x \\0 & \beta & c_y \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} f_x & 0 & 0 & 0\\ 0 & f_y & 0 & 0\\ 0 & 0 & 1 & 0\end{bmatrix}
\begin{bmatrix} X_{c} \\  Y_{c} \\ Z_{c} \\ 1 \end{bmatrix}
\\ &= 
\begin{bmatrix} \alpha f_x & 0 & c_x & 0\\ 0 & \beta f_y & c_y & 0\\ 0 & 0 & 1 & 0\end{bmatrix}
\begin{bmatrix} X_{c} \\  Y_{c} \\ Z_{c} \\ 1 \end{bmatrix}
\end{aligned}
$$

```python
ones = torch.ones((pos.shape[0], pos.shape[1], 1)).to(pos.device)
pos_homo = torch.cat((pos, ones), -1)   # (1, N, 4), each is [x,y,z,1]
projected = torch.bmm(M, pos_homo.permute(0, 2, 1))     # (1, 3, 4) @ (1, 4, N)
projected = projected.permute(0, 2, 1)  # (1, N, 3)
proj = torch.zeros_like(projected)
# u = x / z
proj[..., 0] = projected[..., 0] / projected[..., 2]
# v = y / z
proj[..., 1] = projected[..., 1] / projected[..., 2]
# 1 没意义，我可以存点别的，比如归一化的 0-1 的z-value深度
clip_space, _ = torch.max(projected[..., 2], 1, keepdim=True)
proj[..., 2] = projected[..., 2] / clip_space
```

##### 3.2.2.4. 综合

![](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062025049.png)


像素坐标的齐次坐标点 $P_{uv}=[u, v]$. 外参，投影自然是 w2c.

三种运算方式：

1. 加法
   
    外参 `[3, 3]` 和 `3`

    $$ Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} = K\left( R\begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} + t \right)$$

2. 世界坐标系的欧式点$P_{w}=[X_{w}, Y_{w}, Z_{w}]$，像素坐标的齐次坐标点 $P_{uv}=[u, v]$

    外参 `[3, 4]`

    $$\begin{aligned}
    Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} 
    &= \begin{bmatrix} \alpha f_x & 0 & c_x\\ 0 & \beta f_y & c_y\\ 0 & 0 & 1\end{bmatrix}
    \begin{bmatrix} R & t\end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \\ &= KTP_w
    \end{aligned}
    $$


3. 世界坐标系的齐次坐标点$P_{w}=[X_{w}, Y_{w}, Z_{w}, 1]$
   
    外参 `[4, 4]`

    $$\begin{aligned}
    Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} 
    &= \begin{bmatrix} \alpha f_x & 0 & c_x & 0\\ 0 & \beta f_y & c_y & 0\\ 0 & 0 & 1 & 0\end{bmatrix}
    \begin{bmatrix} R & t \\ 0^T & 1  \end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \\ &= KTP_w
    \end{aligned}
    $$


**相机深度**$z_{c}$ 乘以 **像素坐标**$P_{uv}$ = **相机内参**K 乘以 **相机外参TR** 乘以 **世界坐标**$P_{w}$

像素坐标系下的一点可以被认为是三维空间中的一条射线， $z_{c}$ 就是像素点在相机坐标系下的深度。




## 4. 反向

$\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \mathbf{K}^{−1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} z$：再乘z才是camera下的xyz。

> 例子: 外参，Inverse project 自然是 c2w

例子：已知，$T_{c2w} = [\mathbf{R}, \mathbf{t}]$， $P_c=[u,v,1]^\top$

则，
$$P_w = \mathbf{R}\mathbf{K}^{−1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} + \mathbf{t} $$

- a target pixel $x\in\mathbf{RP}^{2}$ , c2w extrinsics $[R | t]$ , intrinsics $K$ .
- ray origin $o=\mathbf{t}$, ray direction $r=\mathbf{R}\mathbf{K}^{−1}[u,v,1]^\top$

例子：已知，$T_{w2c} = [\mathbf{R}, \mathbf{t}]$， $P_c=[u,v,1]^\top$

则，

$$P_w = \mathbf{R}^{\top}\mathbf{K}^{−1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} + (-\mathbf{R}^\top\mathbf{t})$$

- a target pixel $x\in\mathbf{RP}^{2}$ , w2c extrinsics $[R | t]$ , intrinsics $K$
- ray origin $o=-\mathbf{R}^\top\mathbf{t}$, ray direction $r=\mathbf{R}^{\top}\mathbf{K}^{−1}[u,v,1]^\top$


## 5. ???
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062025209.png)

- $R_c$代表的意思
- w2c和c2w的真正理解还没有理解
  
  第三行：从第四行ray direction的计算来看，原来的extrinsice应该是w2c

  公式1的$R_c$应该是3x3的矩阵，但是第一列是什么意思？怎么会是两个向量的out product?
  
  那么公式1的计算结果，应该是world的坐标，公式2再转化为c2w。

  那么We apply T to every camera pose，这个 camera pose 是不是上面的w2c的extrinsice，把c2w的T乘以camera pose？？？而且是谁先乘谁????