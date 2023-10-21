signal representations

## 维度：

1D signal

3D signal: RGB images

4D signal: radiance fields(color and density)

## 频率

high(fine details of the signal ) and low (smooth signal) frequencies

## 载体：
- polynomials, 
- MLPs: 
    
    Occupancy Networks [35], DeepSDF [40] and NeRF [36]. 

    While MLPs excel in compactness and induce a useful smoothness bias, they are slow to evaluate and hence increase training and inference time.

- 2D and 3D(voxel) feature grids
    
    - voxel grids
        
        《DVGO》
    
        While voxel grids are fast to optimize, they increase memory significantly and do not easily scale to higher dimensions.

- 1D feature vectors

    InstantNGP [37] proposes a hash function in combination with 1D feature vectors, and TensoRF [10] decomposes the signal into matrix and vector products.