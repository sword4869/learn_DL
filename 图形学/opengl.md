3d-gaussian-splatting

```bash
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
```

## 安装opengl
```bash
sudo apt update
sudo apt install build-essential
```
安装OpenGL Library
```bash
# libgl-dev包括libgl1-mesa-dev
sudo apt install libgl-dev
```
glu (OpenGL Utilities)
```bash
sudo apt install libglu1-mesa-dev
```
freeglut3 (OpenGL Utility Toolkit)
```bash
sudo apt install freeglut3-dev
```
glew 

Technically OpenGL is just a specification, implemented by your graphics driver. There's no such thing like a OpenGL SDK library. There's just libGL.so coming with your driver. To use it, you need bindings for your programming language of choise. If that is C, the "bindings" consist of just the header files. However you'll probably also want to use OpenGL extensions, which a easiest used using GLEW.

So I suggest you install the GLEW development files, all the other dependencies (including the OpenGL headers) will be pulled in by the package manager:

```bash
sudo apt install libglew-dev
```

glx

```bash
libgl1-mesa-glx
```
glfw3
```bash
sudo apt install libglfw3-dev
```

## version

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
```
- 核心模式: `OpenGL core profile version string: 4.2 (Core Profile)`, 4.2
- 兼容模式：`OpenGL version string: 4.2 (Compatibility Profile)`, 4.2

