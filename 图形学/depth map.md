disparity map 视差图，depth map 深度图, acc map 

---

depth map 深度图: z-value挂钩，相机look at, 越近数值越接近0，越远越接近1，所以越近越黑，越远越白。

disparity map: 反过来，那么越近越白，越远越黑。

## 体积渲染

acc map就是 weight求和，看看是不是1.

$\text{disparity} = 1 / \text{depth}$

当alpha-blending时， $\displaystyle \hat{D}(\mathbf{r}) =\sum_{i=1}^N \omega_i z_i =\sum_{i=1}^N T_i \alpha_i z_i$
```
depth_map = torch.sum(weights * z_vals, -1)
acc_map = torch.sum(weights, -1)
disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.max(1e-10 * torch.ones_like(acc_map), acc_map))
```

这里有两种z：第一种是nerf使用的深度【存疑，不太确定】，第二种是重投影使用的深度，重投影在uv坐标到相机坐标中用到的是z-value。
- 光线方向的, the leght between the points on the ray and the origin of the ray：$\mathbf{x}_{ij}^k=\mathbf{o}_t+z_k\bar{\mathbf{r}}_{ij}$
- z-value in camera space。

My gt depth is a disparity map. Have you processed it? Otherwise, the current supervision is just opposite to the optimization goal.


```python
import pylab as plt

def depth_to_rgb(depth_map):   
    depth_map = (depth_map - np.min(depth_map)) / np.ptp(depth_map)
    cm = plt.get_cmap('plasma')
    pixel_colored = np.uint8(np.rint(cm(depth_map) * 255))[:, :, :3]
    return Image.fromarray(pixel_color
```