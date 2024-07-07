- [1. 3d-gaussian-splatting](#1-3d-gaussian-splatting)
- [2. 安装opengl](#2-安装opengl)
  - [2.1. 基础](#21-基础)
  - [2.2. 汇总](#22-汇总)
    - [2.2.1. 安装OpenGL Library](#221-安装opengl-library)
    - [2.2.2. glu (OpenGL Utilities)](#222-glu-opengl-utilities)
    - [2.2.3. freeglut3 (OpenGL Utility Toolkit)](#223-freeglut3-opengl-utility-toolkit)
    - [2.2.4. glew](#224-glew)
    - [2.2.5. glfw3](#225-glfw3)
    - [2.2.6. glx](#226-glx)
    - [2.2.7. glm](#227-glm)
- [3. version](#3-version)


---
## 1. 3d-gaussian-splatting

```bash
# Dependencies
sudo apt install -y 
libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
libglew-dev libglfw3-dev 
```

## 2. 安装opengl

https://gist.github.com/Mluckydwyer/8df7782b1a6a040e5d01305222149f3c

### 2.1. 基础
```bash
sudo apt update
sudo apt install build-essential
```
### 2.2. 汇总
glu, glut
```bash
sudo apt install mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev
```

#### 2.2.1. 安装OpenGL Library
```bash
# libgl-dev包括libgl1-mesa-dev
sudo apt install libgl-dev
```
#### 2.2.2. glu (OpenGL Utilities)
```bash
sudo apt install libglu1-mesa-dev
```
#### 2.2.3. freeglut3 (OpenGL Utility Toolkit)
```bash
sudo apt install freeglut3-dev
```
#### 2.2.4. glew 

Technically OpenGL is just a specification, implemented by your graphics driver. There's no such thing like a OpenGL SDK library. There's just libGL.so coming with your driver. To use it, you need bindings for your programming language of choise. If that is C, the "bindings" consist of just the header files. However you'll probably also want to use OpenGL extensions, which a easiest used using GLEW.

So I suggest you install the GLEW development files, all the other dependencies (including the OpenGL headers) will be pulled in by the package manager:

```bash
sudo apt install libglew-dev
```
#### 2.2.5. glfw3
```bash
sudo apt install libglfw3-dev
```
#### 2.2.6. glx

```bash
sudo apt install libgl1-mesa-glx
```
#### 2.2.7. glm

```bash
sudo apt install libglm-dev
```

## 3. version
```bash
sudo apt install mesa-utils
```
```bash
$ glxinfo | grep OpenGL
OpenGL vendor string: Microsoft Corporation
OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 2080 Ti)
OpenGL core profile version string: 4.2 (Core Profile) Mesa 23.0.4-0ubuntu1~22.04.1
OpenGL core profile shading language version string: 4.20
OpenGL core profile context flags: (none)
OpenGL core profile profile mask: core profile
OpenGL core profile extensions:
OpenGL version string: 4.2 (Compatibility Profile) Mesa 23.0.4-0ubuntu1~22.04.1
OpenGL shading language version string: 4.20
OpenGL context flags: (none)
OpenGL profile mask: compatibility profile
OpenGL extensions:
OpenGL ES profile version string: OpenGL ES 3.1 Mesa 23.0.4-0ubuntu1~22.04.1
OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.10
OpenGL ES profile extensions:

$ glxinfo -B
name of display: :0
display: :0  screen: 0
direct rendering: Yes
Extended renderer info (GLX_MESA_query_renderer):
    Vendor: Microsoft Corporation (0xffffffff)
    Device: D3D12 (NVIDIA GeForce RTX 2080 Ti) (0xffffffff)
    Version: 23.0.4
    Accelerated: yes
    Video memory: 27374MB
    Unified memory: no
    Preferred profile: core (0x1)
    Max core profile version: 4.2
    Max compat profile version: 4.2
    Max GLES1 profile version: 1.1
    Max GLES[23] profile version: 3.1
OpenGL vendor string: Microsoft Corporation
OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 2080 Ti)
OpenGL core profile version string: 4.2 (Core Profile) Mesa 23.0.4-0ubuntu1~22.04.1
OpenGL core profile shading language version string: 4.20
OpenGL core profile context flags: (none)
OpenGL core profile profile mask: core profile

OpenGL version string: 4.2 (Compatibility Profile) Mesa 23.0.4-0ubuntu1~22.04.1
OpenGL shading language version string: 4.20
OpenGL context flags: (none)
OpenGL profile mask: compatibility profile

OpenGL ES profile version string: OpenGL ES 3.1 Mesa 23.0.4-0ubuntu1~22.04.1
OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.10
```
- 核心模式: `OpenGL core profile version string: 4.2 (Core Profile)`, 4.2
- 兼容模式：`OpenGL version string: 4.2 (Compatibility Profile)`, 4.2

齿轮测试：弹出绘制齿轮的窗口
```bash
$ glxgears
568 frames in 5.0 seconds = 113.534 FPS
500 frames in 5.0 seconds = 99.805 FPS
X connection to :0 broken (explicit kill or server shutdown).
```
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062033473.png)

