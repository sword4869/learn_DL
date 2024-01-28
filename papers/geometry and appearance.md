volumetric rendering is unable to properly learn scene geometry with **low viewpoint diversity**, and that accurate view synthesis can be obtained with degenerate geometries.

## 稀疏重建

如何解决2D图片到3D的病态映射？通过满足所有图片的颜色损失来构建几何（
It guides the splats to construct a consistent geometry by imposing a constraint on the splats to **satisfy multiple images at the same time**）。

然而 multi-view color supervision 是 local structure, 只有数量够多的才能提供global geometric cue. 增加图片数量就是让局部几何够多，相当于全局几何。

所以就会有图片数量越少，overfitting的问题（局部几何最优，但全局几何很糟，几何一致性越差）

思路就是引入颜色损失之外的损失来提供几何线索。如深度图