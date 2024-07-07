## load_obj

```
# this is a comment
v 1.000000 -1.000000 -1.000000			#  顶点的坐标值，个数是 V
v 1.000000 -1.000000 1.000000
v -1.000000 -1.000000 1.000000
v -1.000000 -1.000000 -1.000000
v 1.000000 1.000000 -1.000000
vt 0.748573 0.750412					# 顶点的uv纹理坐标值，个数是 T: T >= V，因为一个顶点可能属于两个面
vt 0.749279 0.501284
vt 0.999110 0.501077
vt 0.999455 0.750380
vn 0.000000 0.000000 -1.000000			# 顶点的法向量，个数是 N
vn -1.000000 -0.000000 -0.000000
vn -0.000000 -0.000000 1.000000
f 5/2/1 1/2/1 4/3/1						
f 5/1/1 4/3/1 2/4/1
# f面，一个面三个顶点，分别表示`v/vt/vn`.
# 5/2/1 describes the first vertex of the first triangle
#- 5: index of vertex [1.000000 1.000000 -1.000000]
#- 2: index of texture coordinate [0.749279 0.501284]
#- 1: index of normal [0.000000 0.000000 -1.000000]
```

v，vt, vn是坐标值，f里的都是索引。

```python
from pytorch3d.io import load_obj, save_obj

# We read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)

# verts: (V,3). 就是obj中的v
# faces: Faces(verts_idx, normals_idx, textures_idx, materials_idx)
    # verts_id, textures_idx, normals_idx: (F, 3)，同上面 f 5/2/1 1/2/1 4/3/1
# aux: Properties(normals, verts_uvs, material_colors, texture_images, texture_atlas)
    # verts_uvs: (T, 2). 就是obj中的vt
	# normals: (N,3). 就是obj中的vn
```

## Meshes

Meshes是一堆对齐的mesh，所以很多操作都是取出batch=1的单个mesh。

```python
from pytorch3d.structures import Meshes


verts, faces, aux = load_obj(trg_obj)

# 用[]就是因为可以输入多个mesh为一个batch
# 顶点的坐标值 v, 面的顶点索引
trg_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
```
属性
```python
#### TexturesUV 
mesh.textures	
texture_image=mesh.textures.maps_padded()		# torch.Size([1, 1024, 1024, 3])
```
函数
```python
####### .verts_packed(): tensor of vertices of shape (sum(V_n), 3)
# 将batch中的多个mesh的顶点，汇总在一起。
vert_offsets_packed = src_mesh.verts_packed()		

###### .offset_verts(vert_offsets_packed): 得到加上的offset的Meshes
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
new_src_mesh = src_mesh.offset_verts(deform_verts)

####### .get_mesh_verts_faces(index): index就是表示batch中的某个mesh，只有一个那就传入0
# 返回 verts (V, 3)和faces (F, 3).
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
save_obj('final_model.obj', final_verts, final_faces)

####### .verts_list(), .faces_list(): 返回batch中的mesh的顶点/面的列表
# .verts_list(): list of tensors of vertices of shape (V_n, 3)
# .faces_list(): list of tensors of faces of shape (F_n, 3)
verts = mesh.verts_list()[0]
verts /= verts.norm(p=2, dim=1, keepdim=True)
faces = mesh.faces_list()[0]
return Meshes(verts=[verts], faces=[faces])
```

```python
from pytorch3d.ops import sample_points_from_meshes

sample_trg = sample_points_from_meshes(trg_mesh, 5000)
```

## save_obj

```python
from pytorch3d.io import load_obj, save_obj

final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
save_obj('final_model.obj', final_verts, final_faces)
```

