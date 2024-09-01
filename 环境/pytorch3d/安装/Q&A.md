[toc]

## cuda问题

cudatoolkit不行。

因为要nvcc来编译，所以对应版本的系统cuda必须安装。

## ImportError: cannot import name '_C' from 'pytorch3d' (/home/lab/Downloads/pytorch3d-main/pytorch3d/__init__.py)

从安装目录中出来，要不然导入是项目中的文件夹



## gcc

不要使用 conda install gcc。否则会出现，gcc用conda安装的，g++用apt安装的，两者版本不一致而导致编译失败。都用apt安装



```bash
$ TORCH_CUDA_ARCH_LIST="7.5" python -m pip install . 
138 | #error -- unsupported GNU version! gcc versions later than 8 are not supported!
            |  ^~~~~
      error: command '/usr/local/cuda/bin/nvcc' failed with exit code 1


$ sudo apt -y install gcc-8
E: 软件包 gcc-8 没有可安装候选
```

### 安装gcc

https://blog.csdn.net/fengyuyeguirenenen/article/details/130395859: gcc5 和 gcc7。这里用后者

因为ubuntu22新而排除了老的gcc，那么就加入老源

```bash
sudo vim /etc/apt/sources.list

# gcc7
deb https://mirrors.aliyun.com/ubuntu/ focal main universe

sudo apt-get update
sudo apt install gcc-7
sudo apt install g++-7
```



### 切换gcc

g++同理

```bash
(py3d) lab@labU:/media/lab/新加卷/git/EXERCISE/pytorch3d$ ls /usr/bin/gcc* -l
lrwxrwxrwx 1 root root  6  8月  5  2021 /usr/bin/gcc -> gcc-11
lrwxrwxrwx 1 root root 23  5月 13  2023 /usr/bin/gcc-11 -> x86_64-linux-gnu-gcc-11
lrwxrwxrwx 1 root root 22  3月 26  2020 /usr/bin/gcc-7 -> x86_64-linux-gnu-gcc-7
lrwxrwxrwx 1 root root  9  8月  5  2021 /usr/bin/gcc-ar -> gcc-ar-11
lrwxrwxrwx 1 root root 26  5月 13  2023 /usr/bin/gcc-ar-11 -> x86_64-linux-gnu-gcc-ar-11
lrwxrwxrwx 1 root root 25  3月 26  2020 /usr/bin/gcc-ar-7 -> x86_64-linux-gnu-gcc-ar-7
lrwxrwxrwx 1 root root  9  8月  5  2021 /usr/bin/gcc-nm -> gcc-nm-11
lrwxrwxrwx 1 root root 26  5月 13  2023 /usr/bin/gcc-nm-11 -> x86_64-linux-gnu-gcc-nm-11
lrwxrwxrwx 1 root root 25  3月 26  2020 /usr/bin/gcc-nm-7 -> x86_64-linux-gnu-gcc-nm-7
lrwxrwxrwx 1 root root 13  8月  5  2021 /usr/bin/gcc-ranlib -> gcc-ranlib-11
lrwxrwxrwx 1 root root 30  5月 13  2023 /usr/bin/gcc-ranlib-11 -> x86_64-linux-gnu-gcc-ranlib-11
lrwxrwxrwx 1 root root 29  3月 26  2020 /usr/bin/gcc-ranlib-7 -> x86_64-linux-gnu-gcc-ranlib-7

```



一种是手动软链接切换

```bash
# 删除原先的软链接
sudo rm /usr/bin/gcc
# 新建gcc-5到gcc的软链接
sudo ln -s /usr/bin/gcc-7 /usr/bin/gcc
```



另一种是 update-alternatives 切换

```bash
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
update-alternatives: 使用 /usr/bin/gcc-7 来在自动模式中提供 /usr/bin/gcc (gcc)
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives: 使用 /usr/bin/gcc-11 来在自动模式中提供 /usr/bin/gcc (gcc)
$ sudo update-alternatives --config gcc
有 2 个候选项可用于替换 gcc (提供 /usr/bin/gcc)。

  选择       路径           优先级  状态
------------------------------------------------------------
* 0            /usr/bin/gcc-11   11        自动模式
  1            /usr/bin/gcc-11   11        手动模式
  2            /usr/bin/gcc-7    7         手动模式

要维持当前值[*]请按<回车键>，或者键入选择的编号：2       
update-alternatives: 使用 /usr/bin/gcc-7 来在手动模式中提供 /usr/bin/gcc (gcc)
$ gcc --version
gcc (Ubuntu 7.5.0-6ubuntu2) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

## -e

-e的无法上传服务器

## TORCH_CUDA_ARCH_LIST

设置 `TORCH_CUDA_ARCH_LIST` 环境变量主要有以下几个好处：

​	编译时间：当你从源代码编译PyTorch或其他依赖CUDA的库时，指定特定的CUDA架构可以减少编译时间。默认情况下，库可能会为所有支持的CUDA架构生成二进制代码，这可能需要相当长的时间。如果你只指定自己实际使用的那些架构，就可以节省大量时间。

​	磁盘空间：预编译的二进制文件可能会占用大量磁盘空间。通过限制要支持的CUDA架构数量，可以减少这些文件所需的存储空间。

​	运行性能：在某些情况下（尤其是在使用老旧GPU时），为特定架构优化过的代码可能比通用代码运行得更快。
​    

```bash
import torch
torch.cuda.get_device_capability()
# (7, 5)		# 对应就是 7.5

TORCH_CUDA_ARCH_LIST="7.5" python -m pip install . 
```