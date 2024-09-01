[toc]



## 常用
```bash
# 打印到控制台
$ pip list
```
```bash
pip install SomePackage            # latest version
pip install SomePackage==1.0.4     # specific version
pip install SomePackage>=1.0.4     # minimum version
```
```bash
# 离线包
pip install xxx.whl

# 当下载老网络中断，就可以直接去下whl，https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp38-cp38-win_amd64.whl
(py3d) PS D:\git\pytorch3d> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Looking in indexes: https://download.pytorch.org/whl/cu121
Collecting torch
  Downloading https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp38-cp38-win_amd64.whl (2474.0 MB)
     ━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.5/2.5 GB 4.5 MB/s eta 0:07:29
ERROR: Exception:
Traceback (most recent call last):
```
```bash
# 超时
--default-timeout=10

# 更新包
-U, --upgrade
# 更新pip
pip install -U pip
```

```bash
# 输出到文件中
$ pip freeze > requirements.txt

# 安装文件中的包列表
# -r, --requirement <file>
$ pip install -r requirement.txt
```

## git包

```bash  
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
```
`-e`: 
- 会在当前目录下创建`src/clip`。也就是说，会下载到`src`文件夹中（所以不要用`src`作为代码文件夹）。`clip`是根据`#egg=clip`得到的。`#egg=clip` 这个随便起，报错会告诉你真正叫做什么名字。
- `-e`会当前项目安装到python环境中，会使用`src/clip/setup.py`来安装`clip`包。

没有`-e`也行，就是不会显示创建`src/clip`

PS：
1. 不加`.git`下载不下来
```bash
pip install -e git+https://github.com/openai/CLIP@main#egg=clip
```

1. 如果实在不行，可以分解开来
```bash
git clone https://github.com/openai/CLIP.git
cd src/clip
pip install .
```
最后一句 `pip install .` 可以替换为[手动打包](./pip打包.md)提到的 `python setup.py install`。