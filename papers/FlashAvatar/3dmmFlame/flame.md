# flame资源

flame官方网站：[FLAME (mpg.de)](https://flame.is.tue.mpg.de/)

FLAME2020最常用

## FLAME2020

https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1

链接：https://pan.baidu.com/s/1ES9nnKyYQbGYFo2NwLGM0A?pwd=bf2i 
提取码：bf2i

![image-20240701214011573](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407012140594.png)

## generic_model.pkl

上面的FLAME2020中

## FLAME_masks.pkl

顶点各自归属哪部分的mask。



https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip

https://github.com/Zielon/MICA/blob/master/data/FLAME2020/FLAME_masks/FLAME_masks.pkl: windows上有pickle的换行符问题`_pickle.UnpicklingError: the STRING opcode argument must be quoted`，需要换行处理。

链接：https://pan.baidu.com/s/1Rtvnv2jrhhyKNbjTzE9SCg?pwd=81w6 
提取码：81w6

![image-20240701213949749](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407012139779.png)

```python
flame_mask_path = "flame/FLAME_masks/FLAME_masks.pkl"
flame_mask_dic = load_binary_pickle(flame_mask_path)

import sys
import pickle
def load_binary_pickle(filepath):
    with open(filepath, 'rb') as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data
```

![image-20240819132210406](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202408191333915.png)

## FLAME_texture.npz

https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=TextureSpace.zip

链接：https://pan.baidu.com/s/1aqa1via8P6-0UwkT3ZOAYw?pwd=qqkx 
提取码：qqkx

## head_template_mesh.obj

uv parametrization

https://github.com/Zielon/MICA/blob/master/data/FLAME2020/head_template.obj

## landmark_embedding.npy

https://github.com/Zielon/MICA/blob/master/data/FLAME2020/landmark_embedding.npy

## BaselFaceModel.tgz

链接：https://pan.baidu.com/s/1sLLZfFPkSLKmEWQMvMB09Q?pwd=5r5o 
提取码：5r5o



## 汇总

- generic_model.pkl：Flame2020	

# Flame参数

300 shape 𝜷

100 expression 𝝍	

6 pose θ	3K+3的向量，包含了K+1个旋转向量。每个关节的三维旋转矢量加上全局旋转向量。



Flame模型: $M(\vec{\beta},\vec{\theta},\vec{\psi}):\Bbb R^{|\vec{\beta}|\times|\vec{\theta}|\times|\vec{\psi}|}\to\Bbb R^{3N}$​

N=5023个顶点，9976个面，K=4个关节（脖子，下吧，两个眼球）

$\vec{\beta}$ ：形状shape参数

$\vec{\theta}$ ：姿态pose参数，$\vec{\theta}\in\Bbb R^{3K+3}$，包含$K+1$个轴角(aix-angle)坐标系的旋转向量

$\vec{\psi}$ ：表情expression参数





其中shape系数β，pose系数θ和表情系数ψ，输出为N个顶点坐标。

$$
\begin{equation}\begin{split}
M(\vec{\beta},\vec{\theta},\vec{\psi})&=W(T_P(\vec{\beta},\vec{\theta},\vec{\psi}),\mathbf{J}(\vec{\beta}),\theta,\mathbf{\mathcal{W}})
\end{split}\end{equation}
$$

- $W(\overline{\mathbf{T}},\mathbf{J},\vec{\theta},\mathbf{\mathcal{W}})$ 标准蒙皮函数。用于旋转关节$\mathbf{J}\in \Bbb{R}^{3K}$附近的顶点$\overline{\mathbf{T}}$，并由混合权重(blendweights) $\mathbf{\mathcal{W}}\in \Bbb{R}^{K\times N}$进行线性平滑。

------

其中$T_P(\vec{\beta},\vec{\theta},\vec{\psi})$表示了形状、姿态、表情相对于网格模板的偏移量。

$$
\begin{equation}\begin{split}
T_P(\vec{\beta},\vec{\theta},\vec{\psi})=\overline{\mathbf{T}} + B_S(\vec{\beta}, \mathbf{\mathcal{S}})+B_P(\vec{\theta}, \mathbf{\mathcal{P}})+B_E(\vec{\psi},\varepsilon)
\end{split}\end{equation}
$$

- $\overline{\mathbf{T}} \in \mathbb{R}^{3N}$ 网格模板/平均网格脸
- $B_S(\vec{\beta}; \mathbf{\mathcal{S}}): \Bbb{R}^{|\vec{\beta}|} \to \Bbb{R}^{3N}$ 形状blendshape函数。解释与identity相关的形状变化
- $B_P(\vec{\theta}; \mathbf{\mathcal{P}}): \Bbb{R}^{|\vec{\theta}|} \to \Bbb{R}^{3N}$ 姿态blendshape函数。解决线性混合蒙皮不能解决的姿态形变问题。
- $B_E(\vec{\psi};\varepsilon): \Bbb{R}^{|\vec{\psi}|} \to \Bbb{R}^{3N}$ 表情blendshape函数，用来捕捉面部表情。

------

由于不同的脸部形状会产生不同的关节位置，所以关节定义成面部形状的函数

$$
\begin{equation}\begin{split}
\mathbf{J}(\vec{\beta};\mathcal{J},\mathbf{\overline{T}},\mathbf{\mathcal{S}}) = \mathcal{J}(\mathbf{\overline{T}}+B_S(\vec{\beta}; \mathbf{\mathcal{S}}))
\end{split}\end{equation}
$$

- $\mathcal{J}$ 稀疏矩阵，定义了如何从网格顶点计算关节位置

------

形状混合：通过线性混合形状建模得到不同对象的形状变化

$$
\begin{equation}\begin{split}
B_S(\vec{\beta}; \mathbf{\mathcal{S}}) = \sum_{n=1}^{|\vec{\beta}|} \beta_n \mathbf{S}_n
\end{split}\end{equation}
$$

- $\vec{\beta}=[\beta_1,\ldots,\beta_{|\vec{\beta}|}]^T$ 形状shape系数
- $\mathbf{\mathcal{S}}=[\mathbf{S}*1,\ldots,\mathbf{S}*{|\vec{\beta}|}]\in\mathbb{R}^{3N\times|\vec{\beta}|}$ 形状正交基，由PCA方法得到

------

姿态混合

$$
\begin{equation}\begin{split}
B_P(\vec{\theta}; \mathbf{\mathcal{P}}) = \sum_{n=1}^{9K} (R_n(\vec{\theta})-R_n(\vec{\theta}^\ast)) \mathbf{P}_n
\end{split}\end{equation}
$$

- $\vec{\theta}^\ast$ zero pose
- $R(\vec{\theta}):\mathbb{R}^{|\vec{\theta}|}\to\mathbb{R}^{9K}$ 是一个从脸部/头部/眼睛姿势向量$\vec{\theta}$到连接关节旋转矩阵的函数
- $\mathbf{P}_n\in\mathbb{R}^{3N}$ 描述了从$R_n$得到的与静态位姿的顶点偏移。
- $\mathcal{P}=[\mathbf{P}*1,\ldots,\mathbf{P}*{9K}]\in\mathbb{R}^{3N\times9K}$ 姿态空间，包含了所有姿态混合

------

表情混合：由线性blendshapes修改得到

$$
\begin{equation}\begin{split}
B_E(\vec{\psi}; \varepsilon) = \sum_{n=1}^{|\vec{\psi}|} \vec\psi_n \mathbf{E}_n
\end{split}\end{equation}
$$

- $\vec{\psi}=[\psi_1,\ldots,\psi_{|\vec{\psi}|}]^T$ 表情expression系数
- $\varepsilon=[\mathbf{E}*1,\ldots,\mathbf{E}*{|\vec{\psi}|}]\in\mathbb{R}^{3N\times|\vec{\psi}|}$ 表情正交基，训练得到

------

形状模板Template shape

注意，形状，姿势和表情blendshape都是模板网格$\mathbf{\overline{T}}$的位移。我们从一个通用的脸模板网格开始，然后从扫描和模型的其余部分学习$\mathbf{\overline{T}}$。



# Code

## [photometric_optimization](https://github.com/HavenFeng/photometric_optimization)

texture space

```
/data
	/head_template_mesh.obj
	/landmark_embedding.npy
```



## [FLAME_PyTorch](https://github.com/sword4869/FLAME_PyTorch)

```
model
├── FLAME2017
│   ├── Readme.pdf
│   ├── female_model.pkl
│   ├── generic_model.pkl
│   ├── male_model.pkl
├── FLAME2023
│   ├── FLAME Readme.pdf
│   ├── flame2023.pkl
│   └── flame2023_no_jaw.pkl
├── flame_model
│   ├── FLAME_sample.ply
│   ├── flame_dynamic_embedding.npy
│   └── flame_static_embedding.pkl
```

链接：https://pan.baidu.com/s/1W5ZOYDnJqSwR33wGQPbVVg?pwd=ok2q 
提取码：ok2q

```bash
python main.py \
--flame_model_path model/FLAME2023/flame2023.pkl \
--static_landmark_embedding_path model/flame_model/flame_static_embedding.pkl \
--dynamic_landmark_embedding_path model/flame_model/flame_dynamic_embedding.npy
```


```
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
Traceback (most recent call last):
  File "/home/lab/project/FLAME_PyTorch/main.py", line 89, in <module>
    pyrender.Viewer(scene, use_raymond_lighting=True)
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/pyrender/viewer.py", line 347, in __init__
    self._init_and_start_app()
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/pyrender/viewer.py", line 995, in _init_and_start_app
    super(Viewer, self).__init__(config=conf, resizable=True,
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/pyglet/window/xlib/__init__.py", line 133, in __init__
    super(XlibWindow, self).__init__(*args, **kwargs)
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/pyglet/window/__init__.py", line 538, in __init__
    context = config.create_context(gl.current_context)
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/pyglet/gl/xlib.py", line 105, in create_context
    return XlibContext(self, share)
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/pyglet/gl/xlib.py", line 127, in __init__
    raise gl.ContextException('Could not create GL context')
pyglet.gl.ContextException: Could not create GL context
```

```bash
sudo mkdir -p  /usr/lib/dri
conda install -c conda-forge gcc
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so
```

## [MICA](https://github.com/Zielon/MICA)

### 安装：

（1）

```
├── FLAME2020
│   ├── generic_model.pkl		# <<<<
│   ├── head_template.obj
│   ├── landmark_embedding.npy
│   └── FLAME_masks
│       ├── FLAME_masks.gif
│       ├── FLAME_masks.pkl		# 替换
│       └── readme
└── pretrained
    └── mica.tar		# <<<<
```

自带的`data\FLAME2020\FLAME_masks\FLAME_masks.pkl`在windows上有换行问题，我们替换为大礼包中的就行。

（2）

```
C:\Users\lab\.insightface
└── models
    ├── antelopev2
    │   ├── 1k3d68.onnx
    │   ├── 2d106det.onnx
    │   ├── genderage.onnx
    │   ├── glintr100.onnx
    │   └── scrfd_10g_bnkps.onnx
    └── buffalo_l
        ├── 1k3d68.onnx
        ├── 2d106det.onnx
        ├── det_10g.onnx
        ├── genderage.onnx
        └── w600k_r50.onnx
```



（3）

pip install.

```
albumentations==1.3.0
cachetools==5.2.0
chumpy==0.70
coloredlogs==15.0.1
contourpy==1.0.6
cycler==0.11.0
cython==0.29.32
easydict==1.10
face-alignment==1.3.5
falcon==3.1.1
falcon-multipart==0.2.0
flatbuffers==22.11.23
fonttools==4.38.0
google-api-core==2.11.0
google-api-python-client==2.69.0
google-auth==2.15.0
google-auth-httplib2==0.1.0
googleapis-common-protos==1.57.0
gunicorn==20.1.0
httplib2==0.21.0
humanfriendly==10.0
imageio==2.22.4
insightface==0.7
joblib==1.2.0
kiwisolver==1.4.4
llvmlite==0.39.1
loguru==0.6.0
matplotlib==3.6.2
mpmath==1.2.1
networkx==2.8.8
numba==0.56.4
oauth2client==4.1.3
onnx==1.13.0
onnxruntime==1.13.1
opencv-python==4.7.0.72
opencv-python-headless==4.6.0.66
packaging==21.3
prettytable==3.5.0
protobuf==3.20.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pydrive2==1.15.0
pyparsing==3.0.9
python-datauri==1.1.0
python-dateutil==2.8.2
pywavelets==1.4.1
pyyaml==6.0
qudida==0.0.4
rsa==4.9
scikit-image==0.19.3
scikit-learn==1.1.3
scipy==1.9.3
sympy==1.11.1
threadpoolctl==3.1.0
tifffile==2022.10.10
tqdm==4.64.1
trimesh==3.16.4
uritemplate==4.1.1
wcwidth==0.2.5
yacs==0.1.8
```

### 输入输出

```
输入
demo
├── input					
│   ├── carell.jpg			# 输入是图片

输出
demo
├── arcface					# 被裁剪到人脸区域的图片
│   ├── carell.jpg
│   ├── carell.npy			# ？？
└── output					# .npy FLAME parameters, .ply mesh,
    ├── carell					
    │   ├── identity.npy
    │   ├── kpt68.npy
    │   ├── kpt7.npy
    │   ├── mesh.obj
    │   └── mesh.ply
```

## [metrical-tracker](https://github.com/Zielon/metrical-tracker)

### 安装

README里提到的BFM texture等不用管。就按照install.sh的内容配置就行。

```
data
├── head_template_color.obj
├── head_template_mesh.mtl
├── head_template_mesh.obj
├── landmark_embedding.npy
├── uv_mask_eyes.png
├── uv_template.obj
└── FLAME2020
    ├── FLAME_masks.pkl
    ├── FLAME_texture.npz
    └── generic_model.pkl
```

### 输入和输出

```
输入
input
├── duda
│   ├── identity.npy		# 这个就是从MICA得到的
│   └── video.mp4			
```

```
输出
input
├── duda
│   ├── bbox.pt     			
│   ├── identity.npy
│   ├── video.mp4
│   ├── images					# 【255】头部区域
│   │   ├── 00001.png
│   │   ├── 00002.png
│   ├── kpt						# 【255】
│   │   ├── 00001.npy
│   │   ├── 00002.npy
│   ├── kpt_dense				# 【255】
│   │   ├── 00001.npy
│   │   ├── 00002.npy
│   └── source					# 【255】源视频帧
│       ├── 00001.png
│       ├── 00002.png
output
└── duda
    ├── canonical.obj			# 标准mesh
    ├── train.log				# 控制台的日志
    ├── video.avi				# video的输出图片的汇总
    ├── camera					# 对第一帧进行相机校准1000步，每隔100步保存 
    │   ├── 00101.jpg
    │   ├── 00201.jpg
    ├── checkpoint				# 【254】供flashAvatar使用
    │   ├── 00000.frame
    │   ├── 00001.frame
    ├── depth					# 【254】深度，0-超过255的值，要转化
    │   ├── 00000.png
    │   ├── 00001.png
    ├── initialization			# 关键帧的结果
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── 00002.jpg
    ├── input					# 【261】头部帧，一部分和input的images一样，另一个部分是相机校准的第一帧的每隔100步的重复原图。
    │   ├── 00000.png
    │   ├── 00001.png
    ├── logs					# tensorboard
    │   └── events.out.tfevents.1719889676.DESKTOP-E4KCNL9.19204.0
    ├── mesh					# 【254】每帧对应的mesh
    │   ├── 00000.ply
    │   ├── 00001.ply
    ├── pyramid					# 第一帧的金字塔的缩放
    │   ├── 0.png
    │   ├── 1.png
    │   ├── 2.png
    │   └── 3.png
    └── video					# 【254】最终会汇总到video.avi
        ├── 00000.jpg
        ├── 00001.jpg
```

在代码中输入视频以25fps取帧`cfg.fps`。





![image-20240702155937623](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407021559730.png)

### 过程

Tracker

​	initialize_tracking（选择开始几帧作为关键帧）

​		只对额外插入的第1帧（设置里keyframes=[0,1,2]，会插入，变成[0,0,1,2]）：optimize_camera（camera文件夹和input文件夹）

​		每帧 optimize_color、checkpoint（initialization文件夹、input文件夹、depth文件夹）

​	optimize_video

​		每帧optimize_frame

​			optimize_color、checkpoint（video、input、depth）

​	output_video

### code

```python
'''tracker.py L186-210'''
frame = {
    'flame': {
        'exp': self.exp.clone().detach().cpu().numpy(),
        'shape': self.shape.clone().detach().cpu().numpy(),
        'tex': self.tex.clone().detach().cpu().numpy(),
        'sh': self.sh.clone().detach().cpu().numpy(),
        'eyes': self.eyes.clone().detach().cpu().numpy(),
        'eyelids': self.eyelids.clone().detach().cpu().numpy(),
        'jaw': self.jaw.clone().detach().cpu().numpy()
    },
    'camera': {
        'R': self.R.clone().detach().cpu().numpy(),
        't': self.t.clone().detach().cpu().numpy(),
        'fl': self.focal_length.clone().detach().cpu().numpy(),
        'pp': self.principal_point.clone().detach().cpu().numpy(),
    },
    'opencv': {
        'R': opencv[0].clone().detach().cpu().numpy(),
        't': opencv[1].clone().detach().cpu().numpy(),
        'K': opencv[2].clone().detach().cpu().numpy(),
    },
    'img_size': self.image_size.clone().detach().cpu().numpy()[0],
    'frame_id': frame_id,
    'global_step': self.global_step
}
```

使用pytorch3d渲染器。

```python
'''tracker_rasterizer.py'''
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings


class TrackerRasterizer(MeshRasterizer):
```

depth也是从里面获取的

```python
'''tracker.py'''
# DEPTH
depth_view = self.diff_renderer.render_depth(vertices, cameras=self.cameras, faces=torch.cat([util.get_flame_extra_faces(), self.diff_renderer.faces], dim=1))
depth = depth_view[0].permute(1, 2, 0)[..., 2:].cpu().numpy() * 1000.0
cv2.imwrite('{}/{}.png'.format(self.depth_folder, frame_id), depth.astype(np.uint16))
```

```python
'''datasets\image_dataset.py'''
payload = {
    'image': image,				# torch.Size([3, 512, 512])
    'lmk': lmks,				# torch.Size([68, 2])
    'dense_lmk': dense_lmks,	# torch.Size([478, 2])
    'shape': shapes				# torch.Size([300])
}
```

```python
'''datasets\image_dataset.py
input\duda\source中的视频帧，在dataset加载时会跳过第一帧(ffmpeg得到的从00001.png开始)，dataset的第一个是第二帧 00002.png
所以少一个
'''
if self.config.end_frames == 0:		# end_frames=0
    self.images = self.images[self.config.begin_frames:]		# begin_frames=1
```



## [DECA](https://github.com/sword4869/DECA)

```python
deca_code_shape = codedict['shape']     # [B, 100], FLAME parameters (shape 𝜷)
deca_code_exp = codedict['exp']         # [B, 50], FLAME parameters (expression 𝝍)
deca_code_pose = codedict['pose']       # [B, 6], FLAME parameters (pose θ)
deca_code_tex = codedict['tex']         # [B, 50], albedo parameters
deca_code_cam = codedict['cam']         # [B, 3], camera 𝒄
deca_code_light = codedict['light']     # [B, 9, 3], lighting parameters l
# 以上就是 236 dimensional latent code.
deca_code_images = codedict['images']   # [B, C, H, W]
deca_code_detail = codedict['detail']   # [B, 128]
```



## [BFM_to_FLAME](https://github.com/TimoBolkart/BFM_to_FLAME)
