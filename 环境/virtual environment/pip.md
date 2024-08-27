[toc]

## 换源


### 永久修改


这是全局效果，系统python、conda的base环境python、conda自创环境下的python都读取同一个配置文件`~/.config/pip/pip.conf`。


#### pip自动
linux和win都通用。

- 设置

新版ubuntu要求使用https源，要注意。
```bash
# linux 直接复制，windows在前面加$, `$url1='https://mirrors.bfsu.edu.cn/pypi/web/simple'`

# 北外（最快）
url1='https://mirrors.bfsu.edu.cn/pypi/web/simple'
url2='https://mirrors.bfsu.edu.cn'

# 中国科技大学	
url1='https://pypi.mirrors.ustc.edu.cn/simple'
url2='https://pypi.mirrors.ustc.edu.cn'

# 清华(有点限速)
url1='https://pypi.tuna.tsinghua.edu.cn/simple'
url2='https://pypi.tuna.tsinghua.edu.cn'

# 阿里云(十分限速)	
url1='https://mirrors.aliyun.com/pypi/simple'
url2='mirrors.aliyun.com'

pip config set global.index-url $url1
pip config set install.trusted-host $url2
```
修改别的源的话直接再打一遍就会覆盖掉。
- 查看效果
```bash
$ pip config list
global.index-url='https://mirrors.bfsu.edu.cn/pypi/web/simple'
install.trusted-host='https://mirrors.bfsu.edu.cn'
```
- 删除
```bash
$ pip config unset global.index-url 
Writing to ~/.config/pip/pip.conf

# 就少了[global]的index-url
$ cat ~/.config/pip/pip.conf
[install]
trusted-host = https://mirrors.bfsu.edu.cn

# 再删install的trusted-host
$ pip config unset install.trusted-host
```
#### 手动

内容如下：
```
$ vim ~/.config/pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```

> Linux下

修改文件`~/.config/pip/pip.conf`


> windows下

修改文件`C:/Users/Admin/AppData/Roaming/pip/pip.ini`


### 临时换源

可以在使用pip的时候加参数`-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`，其中`-i`(`--index-url`)表示换源，`--trusted-host`表示信任源。不然会有warning，

![20200602155038752](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202408261855580.png)

例如：
```bash
$ pip install pyspider -i $url1 --trusted-host $url2
```
## 命令

### 常用
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

### git包

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