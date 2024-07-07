$q(x)$ 是指diffusion process.

$q(x_t|x_0)$ 就是 $x_0$ 得到 $x_t$ 的过程。
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016186.png)

## $q(x_t|x_0)$

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016187.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016188.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016189.png)

## objective function

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016190.png)

第二项和网络没有关系，$P(x_T)$是从高斯分布取噪音的，定值；$q(x_T|x_0)$ 是什么，*为什么说和网络没有关系？*

第三项有关。$P(x_{t-1}|x_t)$，*这是什么*；$q(x_{t-1}|x_t,x_0)$怎么算

- 你会算的
  
    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016186.png)

- $q(x_{t-1}|x_t,x_0)$ 不是联合分布 $q(x_{t-1}|x_t)$ 和 $q(x_0)$，而是 条件是 $x_t,x_0$ 的条件分布 $q(x_{t-1})$
- ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016191.png)
- 其也是高斯分布
  
    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016192.png)

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016193.png)

    由于 $q(x)$ diffusion process 的 mean variance 是固定的，P(x) denoising process 的 variance 是固定的。所以只需要让 denoising process 的 mean 接近。
    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016194.png)

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016195.png)

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016196.png)


![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016197.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016198.png)

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062016199.png)