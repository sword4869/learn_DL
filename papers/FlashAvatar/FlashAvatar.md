## Test

输入和输出

```
输入
├── dataset
│   └── Obama
│       ├── alpha				# rvm【2401】
│       │   ├── 00000.jpg
│       │   ├── 00001.jpg
│       ├── imgs				# 【2401】
│       │   ├── 00000.jpg	
│       │   ├── 00001.jpg
│       ├── log
│       │   └── ckpt
│       │       └── chkpnt.pth		# 本模型训练的结果
│       └── parsing				# 【2401】【2401】
│           ├── 00000_mouth.png
│           ├── 00000_neckhead.png
│           ├── 00001_mouth.png
│           ├── 00001_neckhead.png

├── metrical-tracker
│   └── output
│       └── Obama
│           └── checkpoint				# metrical-tracker【2400】，表情、相机重演
│               ├── 00000.frame
│               ├── 00001.frame
```

在代码中alpha、imgs、parsing都是跳过第一帧 `frame_delta` 。

```
输出
├── dataset
│       ├── log
│       │   ├── test.avi
```

```bash
# python test.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
python test.py --idname Obama --checkpoint dataset/Obama/log/ckpt/chkpnt.pth
```



## loss

```python
loss = loss_huber*1 + loss_G*1
```

loss_huber是头和嘴的huber。

loss_G是头的lpips。

### huber 损失

特别是在处理带有异常值的数据时。Huber损失相比于平方误差损失 MSE 对异常值更加鲁棒。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190507161515821.png)

当误差小 |y−f(x)| ≤ δ 时，变为 MSE；当误差大 |y−f(x)| > δ 时，则变成类似于 MAE

![img](https://img-blog.csdnimg.cn/img_convert/6354ae93b9172c8ff75f7b4ae81b511c.png)

```python
def huber_loss(network_output, gt, alpha):
    diff = torch.abs(network_output - gt)
    mask = (diff < alpha).float()
    loss = 0.5*diff**2*mask + alpha*(diff-0.5*alpha)*(1.-mask)
    return loss.mean()
```

## 训练

### 数据预处理

#### [MICA](https://github.com/Zielon/MICA)

放入全身的第一帧 `demo\input\duda.jpg`

得到

```
demo\output\duda\identity.npy		# 身份
```



#### [metrical-tracker](https://github.com/Zielon/metrical-tracker)

放入

```
input\duda\identity.npy		# 身份
input\duda\video.mp4     	# 全身的视频
```

得到  

```
input\duda\source 			# 25fps的人头裁剪
output\duda\checkpoint		# .frame文件
```



#### [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)

放入 `input\duda\source`

得到

```
test_res\chosen_merge_00000.png		# neck和head
test_res\chosen_merge_00001.png
```

RVM的效果差，不如就用这个

```
test_res\chosen_merge_00000.png		# 反选背景
test_res\chosen_merge_00001.png
```

### train

```bash
# python train.py --idname <id_name>
python train.py --idname Obama
```

直接写入到 `cv2.VideoWriter`中，没有保留临时图片。



## Code



```python
# train
Scene_mica(data_dir, mica_datadir, train_type=0...)
# test
Scene_mica(data_dir, mica_datadir, train_type=1...)

# test和eval会保留最后的帧，但会test和eval是重叠的
if train_type == 0:
    range_down = 0
    range_up = train_num		# train_num = min(max_train_num, self.N_frames - test_num)
if train_type == 1:
    range_down = self.N_frames - test_num		# test_num = 500
    range_up = self.N_frames
if train_type == 2:
    range_down = self.N_frames - eval_num		# eval_num = 50
    range_up = self.N_frames
```

```
codedict
- shape: torch.Size([1, 300])		# identity, 这个shape在flashAvater是公用的, 采用 00000.frame
- expr: torch.Size([1, 100])		# tracked expression coefficients
- eyes_pose: torch.Size([1, 12])
- eyelids: torch.Size([1, 2])
- jaw_pose: torch.Size([1, 6])

'''tracker.py L186-210'''
frame = {
    'flame': {
        'exp': self.exp.clone().detach().cpu().numpy(),				# codedict['expr'] = viewpoint_cam.exp_param.to("cuda")
        'shape': self.shape.clone().detach().cpu().numpy(),			# 公用的shape: codedict['shape'] = scene.shape_param.to("cuda")
        'tex': self.tex.clone().detach().cpu().numpy(),			
        'sh': self.sh.clone().detach().cpu().numpy(),
        'eyes': self.eyes.clone().detach().cpu().numpy(),			# codedict['eyes_pose'] = viewpoint_cam.eyes_pose.to("cuda")
        'eyelids': self.eyelids.clone().detach().cpu().numpy(),		# codedict['eyelids'] = viewpoint_cam.eyelids.to("cuda")
        'jaw': self.jaw.clone().detach().cpu().numpy()				# codedict['jaw_pose'] = viewpoint_cam.jaw_pose.to("cuda")
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

### DeformModel

```python
# 输出10维
verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)       # torch.Size([1, 13453, 3]), torch.Size([1, 13453, 4]), torch.Size([1, 13453, 3])
```

```python
def decode(self, codedict):
  	# 4个组成ψ
	condition = torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1)		# [1, 120]
    condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)    # torch.Size([1, 14876, 120])
	
    # r(μ_T) 是 uv_vertices_shape_embeded
    uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)	#     torch.Size([1, 14876, 171])
    
    # MLP( r(μ_T), ψ )
    deforms = self.deformNet(uv_vertices_shape_embeded_condition)
```

MLP的输入和输出都是固定维度的...uv_vertices的个数。

​	将表情变量 扩展为  uv_vertices_shape_embeded 的个数。[1, 120] → [1, 14876, 120]

​	输入=cat(表情变量, uv_vertices_shape_embeded)，[1, 14876, 171]

### shape

```python
ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')
payload = torch.load(ckpt_path)
flame_params = payload['flame']
self.shape_param = torch.as_tensor(flame_params['shape'])


codedict['shape'] = scene.shape_param.to("cuda")		# torch.Size([1, 300])


###### 用处1 example_init
DeformModel.example_init(codedict)

###### 用处2 decode
# forward_geo中根据这些参数进行 lbs, 得到变换后的顶点
# torch.Size([1, 5023, 3])
geometry = self.flame_model.forward_geo(
    shape_code,
    expression_params=expr_code,
    jaw_pose_params=jaw_pose,
    eye_pose_params=eyes_pose,
    eyelid_params=eyelids,
)
```

```python
    def decode(self, codedict):
        '''
        Return:
            verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
           # torch.Size([1, 13453, 3]), torch.Size([1, 13453, 4]), torch.Size([1, 13453, 3])
        '''
        ###### 输入：表情
        shape_code = codedict['shape'].detach()
        expr_code = codedict['expr'].detach()
        jaw_pose = codedict['jaw_pose'].detach()
        eyelids = codedict['eyelids'].detach()
        eyes_pose = codedict['eyes_pose'].detach()
        batch_size = shape_code.shape[0]
        condition = torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1)		# torch.Size([1, 120])

        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)		# torch.Size([1, 14876, 120])
        # self.uv_vertices_shape_embeded: torch.Size([1, 14876, 51])
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)	# torch.Size([1, 14876, 171])
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        
        # MLP 输出: uv_vertices_deforms, rot_delta_0, scale_coef
        deforms = torch.tanh(deforms)		# torch.Size([1, 14876, 10])
        uv_vertices_deforms = deforms[..., :3]		# torch.Size([1, 14876, 3])
        rot_delta_0 = deforms[..., 3:7]				# torch.Size([1, 14876, 4])
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)	# torch.Size([1, 14876, 4])
        scale_coef = deforms[..., 7:]		# torch.Size([1, 14876, 3])
        scale_coef = torch.exp(scale_coef)

        # lbs变形
        geometry = self.flame_model.forward_geo(
            shape_code,
            expression_params=expr_code,
            jaw_pose_params=jaw_pose,
            eye_pose_params=eyes_pose,
            eyelid_params=eyelids,
        )
        # 面部顶点的坐标数据： 10006个面，一个面的3个顶点，顶点的坐标
        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))	# torch.Size([1, 10006, 3, 3])  
        # rasterize face_vertices to uv space
        D = face_vertices.shape[-1] # 3
        attributes = face_vertices.clone()		# # torch.Size([1, 10006, 3, 3])
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])	# `torch.Size([10006, 3, 3])

        ###### idx索引矩阵: 像素对应的面
        # 这里的N, H, W, K分别代表了批次大小、高度、宽度和每个像素点对应的顶点数，而D代表了顶点数据的维度。
        N, H, W, K, _ = self.bary_coords.shape	# torch.Size([1, 128, 128, 1, 3])
        # pix_to_face_ori: torch.Size([1, 128, 128, 1])		每个像素点对应的面
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)	# torch.Size([16384, 3, 3])		128*128=16384
        # attributes.gather(0, idx) 根据idx中的索引，从attributes的第0维（即不同面之间）得到像素点对应的面的对应顶点的坐标。
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)		# torch.Size([1, 128, 128, 1, 3, 3])
        # self.bary_coords: torch.Size([1, 128, 128, 1, 3])   3个顶点的ijk
        # pixel_vals 是三维面上的重心的三维坐标：对三个顶点的坐标进行ijk，再汇总出重心的三维坐标
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)	# torch.Size([1, 128, 128, 1, 3])
        # pixel_vals[:, :, :, 0].shape： torch.Size([1, 128, 128, 3])
        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)		# torch.Size([1, 3, 128, 128])
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1		# torch.Size([16384, 3])
        # self.uvmask_flaten_idx: torch.Size([16384])  uv像素对应的面是否属于人头 True,False
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)	# torch.Size([1, 14876, 3])
	
    	########## MLP的坐标输出是对应到uv_vertices的
        # 根据表情变形后的基准点 + MLP根据表情得到offset
        verts_final = uv_vertices + uv_vertices_deforms			# torch.Size([1, 14876, 3])

        # conduct mask：只要人头的
        # self.uv_head_idx: torch.Size([14876])
        verts_final = verts_final[:, self.uv_head_idx, :]		# torch.Size([1, 13453, 3])
        rot_delta = rot_delta[:, self.uv_head_idx, :]			# torch.Size([1, 13453, 4])
        scale_coef = scale_coef[:, self.uv_head_idx, :]			# torch.Size([1, 13453, 3])

        return verts_final, rot_delta, scale_coef
```



```python
def example_init(self, codedict):
		uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1       
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        self.uv_vertices_shape = uv_vertices_shape # for cano init			# torch.Size([1, 14876, 3])
        self.uv_vertices_shape_embeded = self.pts_embedder(uv_vertices_shape)		# torch.Size([1, 14876, 51])
        self.v_num = self.uv_vertices_shape_embeded.shape[1]		# 14876
```

```python
verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)       # torch.Size([1, 13453, 3]), torch.Size([1, 13453, 4]), torch.Size([1, 13453, 3])

if iteration == 1:
    # 高斯初始化的是移动后的点
    gaussians.create_from_verts(verts_final[0])
    gaussians.training_setup(opt)
```

```python
class Deform_Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        mica_flame_config = parse_args()
        self.flame_model = FLAME_mica(mica_flame_config).to(self.device)
        self.default_shape_code = torch.zeros(1, 300, device=self.device)
        self.default_expr_code = torch.zeros(1, 100, device=self.device)
        
        # positional encoding
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)
        
		# aux.verts_uvs: torch.Size([5150, 2]), 每个顶点的uv坐标 (T, 2) T >= V，因为一个顶点可能属于两个面
        _, faces, aux = load_obj('flame/FlameMesh.obj', load_textures=False)
        uv_coords = aux.verts_uvs[None, ...]
        uv_coords = uv_coords * 2 - 1
        uv_coords[..., 1] = - uv_coords[..., 1]		# torch.Size([1, 5150, 2])
        self.uvcoords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0. + 1.], -1).to(self.device)		# torch.Size([1, 5150, 3])
        # faces.textures_idx： 每个面上的纹理索引，This can be used to index into verts_uvs
        self.uvfaces = faces.textures_idx[None, ...].to(self.device)	# torch.Size([1, 10006, 3])
        # faces.verts_idx： 每个面上的顶点索引
        self.tri_faces = faces.verts_idx[None, ...].to(self.device)		# torch.Size([1, 10006, 3])
        
        # rasterizer
        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
        
        # flame mask
        flame_mask_path = "flame/FLAME_masks/FLAME_masks.pkl"   
        flame_mask_dic = load_binary_pickle(flame_mask_path) 
        boundary_id = flame_mask_dic['boundary']
        full_id = np.array(range(5023)).astype(int)
        neckhead_id_list = list(set(full_id)-set(boundary_id))
        self.neckhead_id_list = neckhead_id_list
        self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)
        self.init_networks(device)
```

## QA

Q: UV纹理图映射到Mesh上的顶点，`example_init`是运行几次？

A：只在初始化中运行一次，而迭代中不运行。



Q:  $\mu_T == \mu_M$?

A: 是又不是。准确来是，它们都是纹理图上的像素点对应在mesh上的面的点的坐标。但是每次迭代时，mesh会随表情变化而变化，故而对应面的点的具体坐标是会变动。但是还是维持原本的对应关系，①初始化时像素点到空白表情对应的面的索引关系，②”纹理图上的像素点对应在mesh上的面的点的坐标”，是面的三个顶点的重心与三个顶点的坐标的结果，重心一直用初始化时的，只是三个顶点的坐标随mesh变化而变化。

简单来说，就是还是那个面，只是面在移动。

MLP的输入是经典mesh上的点的坐标 $\mu_T$ ，而迭代中的点的位置由MLP输出的offset和随表情变化的mesh上的点的坐标 $\mu_T$ 相加而成。

也就是说，由表情变化引起的位置变形基本已由 $\mu_M$ 建模（FLame网格很强），MLP输出的offset只是起一些细节作用。
