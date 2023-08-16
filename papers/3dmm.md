一个三维可形变人脸模型（即3D Morphable models），它的核心思想就是可以定义一组人脸作为基底人脸，然后任意一个人脸都可以由这组基底人脸线性加权得到。（关于3DMM的详细细节可以参考知乎大V的这一篇文章：https://zhuanlan.zhihu.com/p/161828142）

1999年文章《A Morphable Model For The Synthesis Of 3D Faces》提出了建立人脸数据库的方法，但没有开源数据集；

Pascal Paysan等人在2009年使用激光扫描仪精确采集了200个人脸数据得到了Basel Face Model数据集，称作BFM数据集；

另一个著名的数据集是2014年提出的FaceWarehouse，不同同样没有开源。

但BFM数据集提供的人脸基地能够表征的人脸表情丰富度十分有限，于是马普所在2017年开源了FLAME，当下最准确、表情最丰富的开源人脸模型。

## BFM

3D Morphable models(简称3DMM)


However, previous works have rarely been a consideration in 3D space since it is hard to obtain accurate 3D coefficients from a single image and the high-quality face render is also hard to design. Inspired by the recent single image deep 3D reconstruction method[2019. Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set]

$$\mathbf{S}=\overline{\mathbf{S}}+\alpha\mathbf{U}_{id}+\beta\mathbf{U}_{exp}$$

- $S$ is the average shape of the 3D face
- $U_{id}$ and $U_{exp}$ are the orthonormal basis of identity and expression of LSFM morphable model
- Coefficients $\alpha \in \R^{80}$ and $\beta \in \R^{64}$ describe the person identity and expression, respectively

To preserve pose variance, coefficients $r \in SO(3)$ and $t \in \R^3$ denote the head rotation and translation. We learn the head pose $\rho = [r, t]$.

To achieve identity irrelevant coefficients generation, we only model the parameters of motion as ${\beta, r, t}$ individually from the driving audio as introduced before.

### Download

Basel Face Model(BFM): 2009 version(最常用), 2019 version.

下载link: [官网](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads), [百度云dqwn](https://pan.baidu.com/s/1-Q0dkJqwE02CWjZQae-giQ?pwd=dqwn)

After getting the access to BFM data, download "**01_MorphableModel.mat**" and put it into ./BFM subfolder.

## FLAME

https://github.com/soubhiksanyal/FLAME_PyTorch/