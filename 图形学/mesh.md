- [mesh](#mesh)
- [ply](#ply)


---
## mesh

https://paulbourke.net/dataformats/obj/

```python
vt 0.823777 0.561993
v -55.4881591796875 50.375640869140625 1044.4334716796875
f 5604/23260 6208/23261 5603/23262
```
- `v`: geometric vertices, 顶点xyz坐标
- `vt`: vertices texture, 纹理uv坐标
- `vn`: vertex normals
- `f`: face, 三角mesh面的三个顶点：顶点id/纹理id

`.obj` 是 index-1 开始的。


```python
def load_obj(filename):
    '''
    load mesh from .obj file.
    
    return: {'uvs', 'verts', 'vert_ids', 'uv_ids'}
    '''
    uvs = []            # vt 0.823777 0.561993
    vertices = []       # v -55.4881591796875 50.375640869140625 1044.4334716796875
    faces_vertex, faces_uv = [], []    # f 5604/23260 6208/23261 5603/23262
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])

    # make sure triangle ids start from index-0
    obj = {
        "uvs": np.array(uvs, dtype=np.float32),
        "verts": np.array(vertices, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }

    # for k, v in obj.items():
    #     print(k, v.shape)
    return obj
```

## ply

```python
ply
format ascii 1.0
comment Created by Open3D
comment this file is a cube
element vertex 20208
property double x
property double y
property double z
property double nx
property double ny
property double nz
property uchar red            
property uchar green
property uchar blue  
element face 39904
property list uchar uint vertex_indices
end_header
75.4157 18.6231 -52.7512 0.670346 0.146898 -0.727363
77.6993 16.5636 -51.896 0.598826 0.273568 -0.752707
...
3 4003 19998 16994
3 19998 5099 19887
```

- `property list uchar uint vertex_indices`, `3 19998 5099 19887`: 三角mesh
    
    这意味着属性“vertex_index"首先包含一个无符号字符，告诉该属性包含多少个索引，然后是一个包含那么多整数的列表。这个可变长度列表中的每个整数都是一个顶点的索引。
```python
import open3d as o3d

o3d.io.write_triangle_mesh(ply_path, mesh, write_ascii=True)
```
- `write_ascii`: True, `format ascii 1.0`; False, `format binary_little_endian 1.0`. ascii可人读，二进制。



```python
from plyfile import PlyData

path = r'D:\git\TODO\000213\face.ply'
plydata = PlyData.read(path)
comments = plydata.comments         # ['Created by Open3D']

### element vertex 
vertex = plydata['vertex']
# count
vertex_count = vertex.count         # 20208
# property
x = vertex['x']                     # <class 'numpy.ndarray'>, shape (20208,)
y = vertex['y']                     # <class 'numpy.ndarray'>, shape (20208,)
vertex_0 = vertex[0]                # <class 'numpy.void'>, (-38.8725, 35.1234, 21.4028, -0.570124, 0.298101, 0.765568)
# vertex_0['x']
# 75.4157
# vertex_0[0]
# 75.4157
# vertex_0.shape
# ()

### element face 
face = plydata['face']
face_count = face.count           # 39904
vertex_indices = face['vertex_indices']     # <class 'numpy.ndarray'>, shape (39904,)
# faces['vertex_indices'][0]
# array([   0, 5118, 5120], dtype=uint32)
```

```
from plyfile import PlyElement

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
```