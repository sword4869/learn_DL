[toc]
# venv & virtualenv

venv:
- Since Python 3.3, `venv` has been integrated into the standard library. So you don't need to install venv module.
- 只能在3.3版本以后，2.x用不了


virtualenv:
- 同时支持 python2 和 python3


venv & virtualenv:
- `venv` is a subset of `virtualenv`. So if you want to use full capacities, you need to install `virtualenv`.
- 但是通常只用`venv`就够了

与conda：结论是不如conda
- venv和virtualenv创建虚拟环境，只能使用当前系统已经安装好的python，**不能指定系统不存在的python环境版本**。而 conda 可以
- broken system python
  venv & virtualenv 只是在系统python下的外壳，其内部几乎没有什么内容，并且大部分都是从系统python中借用的（这就是您在安装python时安装的内容）。这意味着，如果你升级你的系统python，你的虚拟环境可能会崩溃，所以要小心。对于系统python来说，这样做的好处是创建虚拟环境可以很快。
  而conda没有此问题。
- 打包
  conda也可以。

# venv command
## create

```bash
# ENVIRONMENT contain path, such as 
# tutorial(current directory),
# ~/hello-world/tutorial(non-current directory)
$ python3.8 -m venv ENVIRONMENT
```
`python -m`不可省略：不同于之后的`virtualenv`是一个可执行文件，`venv`只是包，所以要`python -m`来执行。

containing a copy of the current Python interpreter, `python3.8`.

`ENVIRONMENT`: When you download modules, the location of installed modules is here.

The `venv` module is often in **project** directory, every project configurates a virtual environment. 

When you want to delete the project, you also delete the directory of this virtual environment. `rm -rf ENVIRONMENT`

## activate & deactivate

```bash
$ source ENVIRONMENT/bin/activate

$ deactivate
```

PS: Don't use hidden direcotor as `ENVIRONMENT`.
In some case, if your `ENVIRONMENT` is `.xxxx`(hidden directory), then source may not correctly work.
```bash
$ source .vv/bin/activate

# even it shows (.vv), it is actually wrong.
# python interpreter should be .vv/bin/python
(.vv)$ which python
/home/admin/anaconda3/bin/python
```
## example

```bash
# get into project directory
$ cd ~/PROJECT

# create a python virtual environment, named hello
$ python -m venv venv

# activate it
# linux
$ source venv/bin/activate
# windows
$ .\venv\Scripts\activate

# please upgrade before the installation of any package.
$ python -m pip install --upgrade pip

$ deactivate
```

# virtualenv command
## install

```bash
$ pip install virtualenv

$ virtualenv --version
virtualenv 20.16.5 from /home/sword/.local/lib/python3.8/site-packages/virtualenv/__init__.py
```


## create

```bash
# virtualenv ENVIRONMENT
$ virtualenv .venv
```
> 继承包

virtualenv从版本20开始，默认就是`--no-site-packages`了，默认就是不继承父环境的包。创建的虚拟环境是一个不带任何第三方包的“干净”的Python运行环境。

如果有教程告诉你，`--no-site-packages`选择不继承父环境的包，这个命令已经废弃了。

那么，如果你现在想要继承父环境的包，怎么办？使用`--system-site-packages`

> python不同版本

```bash
# 指定路径
$ virtualenv --python=/usr/bin/python3.8 .venv

# 自动寻找系统上已安装的python3.8位置
$ virtualenv --python=3.8 .venv

# 不能安没有的python
$ virtualenv --python=3.9 .venv
RuntimeError: failed to find interpreter for Builtin discover of python_spec='3.9'
```
# other



## pip 
`pip install --user package_name`

使用`--user`仅安装在当前用户的路径下，其他用户看不见；虚拟环境中无需使用该配置，因为用户目录在该环境下不可见。

## 在脚本中指定编译器


```python
#!/usr/bin/env python
```
这种写法在你机器上安装了多个版本的python的时候有意义，这样声明的时候，会去取你机器的 PATH 中指定的第一个 python 来执行你的脚本。如果这时候你又配置了虚拟环境的话，那么这样写可以保证脚本会使用你虚拟环境中的 python 来执行。

而不是`#!/usr/bin/python`，写死了，就是要 `/usr/bin/python`的 python 来执行你的脚本。