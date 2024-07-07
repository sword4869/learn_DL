多视角的视频（随时间变换身体姿势）

## 标准空间+变形

使用LSB来deform

## 标准空间

1. 每个高斯：5（位置、方向、尺寸、颜色、不透明度）+2（**可学习的蒙皮权重、可学习的非刚性形变的编码**）

2. SMPL初始化。body shape β and body poses θt = (θ1, θj, . . . , θJ )。
   
    高斯中心设置为顶点位置

    在训练过程中，**高斯数量会增减**。

## pose 为条件的 deform，刚性+局部非刚性

1. 高斯的尺寸和不透明度不动 across novel views and novel poses。只变位置和旋转。

2. 刚性
   
    给定身体姿势 θt → 计算出各身体关节的刚性变换 Mj → 得到每个高斯的蒙皮变换 T，为根据每个高斯的蒙皮权重乘以其各身体关节的刚性变换 Mj

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014737.png)

    → 以 T 来变换高斯的位置和方向。

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014738.png)

    - 学的蒙皮权重，而不是直接copy SMPL的权重。

        ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014739.png)

        ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014740.png)

3. 非刚性残差

    为什么需要？因为要建模包括**衣服**的非刚性形变部分。

    MLP
    - 输入：人体姿 θ t与高斯的可学习编码 l。这个可学习编码就负责学习局部变形。
    - 输出：平移向量tmlp，方向向量qmlp的四元数、环境遮挡比例因子s。

        ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014741.png)
        
        为什么Rt？因为为了能够泛化，残差平移向量被定义在标准空间中，因此需要随着LBS的方向旋转。

        参照图形管道中的环境遮挡，服装褶皱引起的自遮挡和阴影。MLP输出比例因子s∈[0，2]，该比例因子将每个高斯的RGB颜色相乘，然后被clip到[0，1]中。

    - 高斯的position→latent code

        我们用每个高斯中心的规范位置的位置编码替换潜在代码。我们的潜在特征嵌入模型在所有指标上都优于位置编码嵌入模型。这是因为在正则空间中每个高斯中心上的位置编码可能会过度约束运动，因为**变形可能不会平滑地与空间位置相关**，而相比之下，我们的具有设计的潜在代码的模型可以自适应地学习有意义的嵌入来细化每个运动。

        ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014742.png)

## loss

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014743.png)

1. Reconstruction losses，即颜色loss
   
   segmentation mask, 只计算前景. 
   
   L1, ssim, lpips

2. 最小化MLP输出：

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014744.png)

    我们希望我们的变形尽可能依赖于LBS，并期望MLP只学习局部变形。因此，我们限制MLP输出尽可能小。
    
    Ltrans控制平移残差的范数，让其接近0；
    Lrot将旋转残差推近**单位**四元数qid；
    Lamb引导环境遮挡因子保持接近1，也就是对颜色不做改变，乘以1。

3. Regularization of canonical positions

    我们鼓励高斯位置保持靠近SMPL网格，以避免浮动伪影。

    ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014745.png)

    我们从每个高斯正则位置pi搜索最近的顶点vi，这将惩罚比阈值（因为我们的模型表示穿着衣服的身体，所以我们定义了一个阈值τ pos来表示皮肤和衣服之间的最大距离）更远的点（RELU消减掉负数）

4. 蒙皮权重监督薄弱：
   
   因为我们在有限数量的手势上训练，所以训练数据通常可以用不能很好概括的蒙皮权重来拟合。高斯函数的蒙皮权重与SMPL的蒙皮权重进行软监控，即高斯函数离顶点越近，其蒙皮权重就需要越相似

   ![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062014746.png)

## other

render an image of size 512x512 in 50ms or 20fps on a single Tesla V100 GPU.

Our avatar models contains usually around 40k gaussians, that corresponds to a memory footprint of 2.2MB per frame that need to be cached in memory.

Training the model for our experiments takes from 5 to 20 hours on a Tesla V100, depending on the dataset size.