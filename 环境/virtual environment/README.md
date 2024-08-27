[toc]

# README
## 问题一：系统python（apt安装的）还是conda的python？


**conda的python**！

理由：

> 系统python没有pip

出现场景的电脑系统：linux。

linux现在不带系统python3，`apt install python3`后发现安的python没有pip。

所以，与其用系统的python，不如直接安装conda，python和pip都有。

> 系统python没有环境管理功能

想要为一个项目创建一个单独的conda环境，python版本和库版本。如果使用系统python的话，就要卸载原来的python库，重新安装。

## 问题二：缓存问题

> conda缓存

conda install把包缓存在`miniconda3/pkgs`下。

> pip缓存

- 查看缓存位置
  ```bash
  pip cache dir
  ```
- 全局禁用缓存
  ```bash
  pip config set install.no-cache-dir true
  ```
- 删除缓存
  ```bash
  # 从 pip 的缓存中删除所有与 matplotlib 相关的轮文件
  pip cache remove matplotlib
  # 从 pip 的缓存中清除所有轮文件。
  pip cache purge
  ```

