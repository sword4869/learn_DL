[toc]

## setuptools

```bash
# setuptools 已经被包含在 bulid 中，不用`pip install --upgrade setuptools`
pip install --upgrade build
```

```python
from setuptools import setup, find_packages

setup(
    name='markmove',
    version='0.0.1'
)
```
[Attributes](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#entry-points)

### 包信息

必选 `name` 和 `version`。

```python
setup(
    name='markmove',
    version="0.0.1",
    description='a pip package which is used to move markdown files',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="sword4869",
    author_email="xxx@qq.com",
    url='https://github.com/sword4869/markmove',
)
```
#### 包名
-  `name`(可以包含`_`,`-`) 是在`pip list`中显示的包名

PS: 在`python -m module_name`和`import module_name`的包名`module_name`不可以包含`-`.

#### 主要信息
-   `version` 是包版本
-   `description` 是一个简短的，一句话的包的总结。
-   `author` 和 `author_email`用于识别包的作者。
-   `maintainer` 维护者 `maintainer_email` 维护者的电子邮件地址
-   `long_description`是包的详细说明。这显示在Python Package Index的包详细信息包中。
    `long_description_content_type`告诉索引什么类型的标记用于长描述。
-   `keywords` 指定包关键字，用于Pypi网站检索。
-   `classifiers`告诉pypi的信息（发布包用的）

    ```python
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # 发展时期,常见的如下
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
    
        # 开发的目标用户
        'Intended Audience :: Developers',
    
        # 属于什么类型
        'Topic :: Software Development :: Build Tools',
    
        # 许可证信息
        'License :: OSI Approved :: MIT License',
    
        # 目标 Python 版本
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    ```
-   `license` 许可证信息
-   `classifiers` 指定包所属分类
-   `url`是项目主页的URL。对于许多项目，这只是一个指向GitHub，GitLab，Bitbucket或类似代码托管服务的链接。
-   `download_url` 指向下载链接
-   `platforms` 指定包所适用操作系统

其他
-   `cmdclass` 扩展命令

```python
from setuptools import Command

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

setup(
    cmdclass={
        'upload': UploadCommand,
    },
)
```
-   `ext_modules` 扩展模块列表，用于构建 C 和 C++ 扩展扩展包
-   `scripts` 脚本列表，会被安装到操作系统相关路径 PATH 路径下

### 包位置

#### packages

##### 一级包

有几个一级packages安装几个一级包。

```python
└── src
    ├── markmove
    │   ├── __init__.py
    │   └── main.py
    └── mypkg
        ├── __init__.py
        ├── q.py
        ├── subpkg1
            ├── __init__.py
        └── subpkg2
            ├── __init__.py

print(find_packages(where='src'))
# ['markmove', 'mypkg', 'mypkg.subpkg1', 'mypkg.subpkg2']
```
```python
>>> import markmove
>>> import mypkg
```



##### 原理
> `packages=find_packages()` 或者 `find_namespace_packages()`

`find_packages()` 忽略不带`__init__.py`的包，而 `find_namespace_packages()` 不忽略不带`__init__.py`的包。

> `find_packages` 的default值
```python
# All keyword arguments below are optional:
# 
packages=find_packages(
    where='.',          # 查找的根目录
    include=['*'],  
    exclude=[]
),
```
> where

e.g.1 因为 `src` 没有 `__init__.py` 而不识别，更别提子包
```
project_root_directory
└── src/
    └── mypkg/
        ├── __init__.py
        ├── subpkg1/
        │   ├── __init__.py
            ├── subsubpkg1/
            │   ├── __init__.py
                ├── subpkg1/
        ├── subpkg2/
        │   ├── __init__.py

print(find_packages())
# [] 啥有没找到
```
e.g.2 识别 `mypkg`，及其子包
```python
project_root_directory
└── src/
    └── mypkg/
        ├── __init__.py
        ├── subpkg1/
        │   ├── __init__.py
            ├── subsubpkg1/
            │   ├── __init__.py

print(find_packages(where='src'))
# ['mypkg', 'mypkg.subpkg1', 'mypkg.subpkg1.subsubpkg1']
```

> include：只留，相当于 where 且 include

所以，`find_packages(include=['xxx'])`中，如果`xxx`包不带`__init__.py`，那这样写也找不到。

```python
print(find_packages(where='src', include=['mypkg.subpkg1']))
# ['mypkg.subpkg1']  
# 注意，这样写是没有子包的 'mypkg.subpkg1.subsubpkg1'

# mypkg.subpkg1*
print(find_packages(where='src', include=['mypkg.subpkg1*']))
# ['mypkg.subpkg1', 'mypkg.subpkg1.subsubpkg1']
```

> exclude: where 后删除 exclude

```python
print(find_packages(where='src', exclude=['mypkg.subpkg1']))
# [mypkg', 'mypkg.subpkg2', 'mypkg.subpkg1.subsubpkg1']
# 注意，这样写仅仅删除父包，其子包并不删除

# mypkg.subpkg1*
print(find_packages(where='src', exclude=['mypkg.subpkg1', 'mypkg.subpkg1.*']))
# ['mypkg', 'mypkg.subpkg2']
```

##### 两种格式
###### src-layout
```bash
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
├── docs
└── src/            # 非包
    └── mypkg/
        ├── __init__.py
        ├── data            # 文件资源在包里，不在外面
        │   └── data1.txt
        ├── module.py
        ├── subpkg1/
        │   ├── __init__.py
        │   ├── ...
        │   └── module1.py
        └── subpkg2/
            ├── __init__.py
            ├── ...
            └── module2.py
```
```python
setup(
    packages=find_packages(
        where='src',
    ),
    package_dir={"": "src"}
)
```
`package_dir` 表示相关代码是包含在特定目录中（而不是直接放在项目的根目录下）

不用`package_dir`会报错。

```
Processing d:\git\markmove
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [9 lines of output]
      running egg_info
      creating C:\Users\lab\AppData\Local\Temp\pip-pip-egg-info-7wpmdp4n\markmove.egg-info
      writing C:\Users\lab\AppData\Local\Temp\pip-pip-egg-info-7wpmdp4n\markmove.egg-info\PKG-INFO
      writing dependency_links to C:\Users\lab\AppData\Local\Temp\pip-pip-egg-info-7wpmdp4n\markmove.egg-info\dependency_links.txt
      writing entry points to C:\Users\lab\AppData\Local\Temp\pip-pip-egg-info-7wpmdp4n\markmove.egg-info\entry_points.txt
      writing requirements to C:\Users\lab\AppData\Local\Temp\pip-pip-egg-info-7wpmdp4n\markmove.egg-info\requires.txt
      writing top-level names to C:\Users\lab\AppData\Local\Temp\pip-pip-egg-info-7wpmdp4n\markmove.egg-info\top_level.txt
      writing manifest file 'C:\Users\lab\AppData\Local\Temp\pip-pip-egg-info-7wpmdp4n\markmove.egg-info\SOURCES.txt'
      error: package directory 'markmove' does not exist
      [end of output]

```
###### flat-layout

```
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
├── docs            # 不是包
└── mypkg/          # 是包
    ├── __init__.py
    ├── data            # 文件资源在包里，不在外面
    │   └── data1.txt
    ├── module.py
    ├── subpkg1/
    │   ├── __init__.py
    │   ├── ...
    │   └── module1.py
    └── subpkg2/
        ├── __init__.py
        ├── ...
        └── module2.py
```
```python
setup(
    packages=find_packages()
)
```

#### py_modules

`py_modules=["six"]`: 意思是 `six.py`
    
If your project contains any single-file Python modules that aren’t part of a package, set `py_modules` to a list of the names of the modules (minus the `.py` extension) in order to make Setuptools aware of them.

```python
setup(
    packages=find_packages(exclude=["tests*"]),
    py_modules=["clip"],
)
```
#### Automatic discovery

Automatic discovery will only be enabled if you don’t provide any configuration for `packages` and `py_modules`. If at least one of them is explicitly set, automatic discovery will not take place. Note: specifying `ext_modules` might also prevent auto-discover from taking place

适合 src-layout, 不适合 flat-layout

> src-layout: Automatic discovery 会自动进入 src 寻找包。无须担心 `package_dir`。
```python
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
├── docs
├── images
└── src/            # Automatic discovery 会自动进入
    └── mypkg/
        ├── __init__.py
        ├── data            # 文件资源在包里，不在外面
        │   └── data1.txt
        ├── module.py
        ├── subpkg1/
        │   ├── __init__.py
        │   ├── ...
        │   └── module1.py
        └── subpkg2/
            ├── __init__.py
            ├── ...
            └── module2.py

from setuptools import find_packages, setup

setup(
    name='markmove',
    version='0.0.3',
)

# 效果一样
>>> import markmove
>>> import mypkg
```

> 在flat-layout下，Automatic discovery根据屏蔽词而屏蔽某些文件夹和py文件，但是有局限而报错。

Reserved package names:
`FlatLayoutPackageFinder.DEFAULT_EXCLUDE: Tuple[str, ...] = ('ci', 'ci.*', 'bin', 'bin.*', 'debian', 'debian.*', 'doc', 'doc.*', 'docs', 'docs.*', 'documentation', 'documentation.*', 'manpages', 'manpages.*', 'news', 'news.*', 'newsfragments', 'newsfragments.*', 'changelog', 'changelog.*', 'test', 'test.*', 'tests', 'tests.*', 'unit_test', 'unit_test.*', 'unit_tests', 'unit_tests.*', 'example', 'example.*', 'examples', 'examples.*', 'scripts', 'scripts.*', 'tools', 'tools.*', 'util', 'util.*', 'utils', 'utils.*', 'python', 'python.*', 'build', 'build.*', 'dist', 'dist.*', 'venv', 'venv.*', 'env', 'env.*', 'requirements', 'requirements.*', 'tasks', 'tasks.*', 'fabfile', 'fabfile.*', 'site_scons', 'site_scons.*', 'benchmark', 'benchmark.*', 'benchmarks', 'benchmarks.*', 'exercise', 'exercise.*', 'exercises', 'exercises.*', 'htmlcov', 'htmlcov.*', '[._]*', '[._]*.*')`

Reserved top-level module names:
`FlatLayoutModuleFinder.DEFAULT_EXCLUDE: Tuple[str, ...] = ('setup', 'conftest', 'test', 'tests', 'example', 'examples', 'build', 'toxfile', 'noxfile', 'pavement', 'dodo', 'tasks', 'fabfile', '[Ss][Cc]onstruct', 'conanfile', 'manage', 'benchmark', 'benchmarks', 'exercise', 'exercises', '[._]*')`

```python
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
├── docs            # Automatic discovery 根据屏蔽词屏蔽docs
├── images          # Automatic discovery 的屏蔽词没有这个
└── mypkg/
    ├── __init__.py
    ├── data            # 文件资源在包里，不在外面
    │   └── data1.txt
    ├── module.py
    ├── subpkg1/
    │   ├── __init__.py
    │   ├── ...
    │   └── module1.py
    └── subpkg2/
        ├── __init__.py
        ├── ...
        └── module2.py

from setuptools import find_packages, setup

setup(
    name='markmove',
    version='0.0.3',
)
'''
Processing d:\git\mytree
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [14 lines of output]
      error: Multiple top-level packages discovered in a flat-layout: ['images', 'mytree'].
     
      To avoid accidental inclusion of unwanted files or directories,
      setuptools will not proceed with this build.
     
      If you are trying to create a single distribution with multiple packages
      on purpose, you should not rely on automatic discovery.
      Instead, consider the following options:
     
      1. set up custom discovery (`find` directive with `include` or `exclude`)
      2. use a `src-layout`
      3. explicitly set `py_modules` or `packages` with a list of names
     
      To find more information, look for "package discovery" on setuptools docs.
      [end of output]
'''
```

### python版本
```python
    python_requires=">=3.8",
```

### 依赖
-   `install_requires`：当前模块依赖哪些包，若环境中没有，则会自动下载安装

    ```python
        install_requires=[
            'torch',            
            'numpy==1.25.1',    
            ...,
            'requests[security, socks] >= 2.18.4',
            # 指定python版本
            "enum34;python_version<'3.4'",
            # 指定平台
            'pywin32 >= 1.0;platform_system=='Windows'',
            # git
            "Package-A @ git+https://example.net/package-a.git"
        ],
    ```

    - 同 [requirements.txt的写法](./requirements.md#1-basic)

    - `'requests[security, socks] >= 2.18.4'`，这是包含了额外的可选依赖`extras_require`的形式。

      - 正常安装requests会自动安装它的`install_requires`中指定的依赖，而不会安装security和socks这两组依赖。 
      - 这两组依赖是定义在它的`extras_require`中。 这种形式，用在深度使用某些库时。


-   `setup_requires` setup.py 本身要依赖的包，这通常是为一些setuptools的插件准备的配置，这里列出的包，不会自动安装。
-   `tests_require` 仅在测试时需要使用的依赖，在正常发布的代码中是没有用的。在执行`python setup.py test`时，自动安装
-   `dependency_links` 用于安装`setup_requires`或`tests_require`里的软件包。这些信息会写入egg的 metadata 信息中
    ```python
        dependency_links=[
            "http://example2.com/p/foobar-1.0.tar.gz",
        ],
    ```
-   `extras_require` 不会在安装模块时会自动安装依赖包，这里仅表示该模块会依赖这些包。但是这些包通常不会使用到，只有当你深度使用模块时，才会用到，这里需要你手动安装
    ```python
        extras_require={
            'PDF':  ["ReportLab>=1.2", "RXP"],
            'reST': ["docutils>=0.3"],
        }
    ```


自动提取 requirements.txt的内容
```python
from setuptools import setup
import os
import pkg_resources

setup(
    install_requires=[
        str(r) for r in pkg_resources.parse_requirements(open(os.path.join(os.path.dirname(__file__), "requirements.txt")))
    ],
    extras_require={'dev': ['pytest']},
)
```
### 可执行脚本

比`python -m module_name`（写了`__main__.py`）还要简洁：直接输入`cli-name`。

```python
    entry_points={
        'console_scripts': [
            'cli-name = mypkg.mymodule:some_func',
        ]
    }
```
syntax：
- `cli-name`:
  - 安装后会生成一个可执行文件：`D:\miniconda\envs\test\Scripts\cli-name.exe`
  - 所以可以直接在命令行执行
    ```bash
    任意位置 $cli-name
    hello, this is the print text of hello() function in module_name/m.py
    ```
- `mypkg.mymodule:some_func` 相当于 `from mypkg.mymodule import some_func`
  
  `mypkg:some_func` 相当于 `from mypkg import some_func`(定义在`__init__.py`中)


PS: 除了命令行脚本 `console_scripts`，还有GUI脚本 [gui_scripts](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#gui-scripts)

### 文件资源
#### 手动指定 package_data
- `package_data={'sample': ['package_data.dat'],}` 包含**在包里面的**数据文件，会被复制到包文件夹中。

    `sample`是包名，`'package_data.dat'`是文件相对于包文件夹的相对位置。

- `exclude_package_data={"mypkg": [".gitattributes"]}`

    The `exclude_package_data` option is a dictionary mapping package names to lists of wildcard patterns, just like the `package_data` option. 
    
    And, just as with that option, you can use the empty string key `""` in `setup.py` to match all top-level packages.

    Any files that match these patterns will be _excluded_ from installation, even if they were listed in `package_data` or were included as a result of using `include_package_data`.

![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202408261856557.png)
#### 最没用 include_package_data
自动扫描和加入在 `packages` 中的包里的文件资源。

前提是：
![Alt text](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202408261856558.png)

```python
from setuptools import setup, find_namespace_packages
setup(
    packages=...,
    include_package_data=True,
)
```

也能配合用 `exclude_package_data`。

#### 已废弃 data_files
`data_files=[('my_data', ['data/data_file'])],`

`data_files` is **deprecated**. It does not work with wheels, so it should be avoided.

手动指定要包含在安装套装里面数据文件

### pypi发布

<https://pypi.org/account/register> 注册账号。

```bash
pip install build
python -m build --wheel .
pip install twine
twine upload dist/*
```

## pip Commands

> `python setup.py install` 将被 `pip install` 取代
```bash
# python setup.py install
pip install .
```

| Old command | New command |
| --- | --- |
| `python setup.py install` | `pip install` |
| `python setup.py develop` | `pip install -e` |
| `python setup.py sdist` 源码 | `python -m build --sdist`|
| `python setup.py bdist_wheel` 轮子| `python -m build --wheel` |
| `python setup.py test` | `pytest` (usually via tox or nox) |
| `python setup.py upload` | `twine upload` (with twine) |
| `python setup.py check` | `twine check` (this doesn't do all the same checks but it's a start) |
| Custom commands | tox and nox environments |

### install

#### install in “development mode”

`pip install` 将包复制一份安装到`envs\test\Lib\site-packages`中。
```bash
$ pip list
Package         Version
--------------- -------
mytree          0.0.1
```

`pip install -e` (`-e` is short for `--editabla`， 代表  “editable/development” mode)：不复制包，直接定位到当前工程处。
```bash
$ pip list
Package        Version Editable project location
-------------- ------- ----------------------------------------
ConfigArgParse 1.7
mytree         0.0.1   d:\git\localtoolkit\linux_scripts\mytree
```

区别就是，当安装为可编辑`pip install -e`时，可以在不重新安装的情况下就地编辑项目，对Python源文件的更改将在下次启动解释器过程时反映出来。而`pip install`编辑后需要重新安装。

#### init文件和 development mode
```bash
project_root_directory
├── setup.py        # and/or setup.cfg, pyproject.toml
└── module_name
    ├── m.py        # 简单的hello()函数
    └── __init__.py # `__init__.py` 决定是否必须 `-e`
```

当没有`__init__.py` 文件是，会出现一个奇特的现象：
- `pip install git+https://github.com/sword4869/mytree.git` 会出现 *ModuleNotFoundError*，
- 而 `pip install -e git+https://github.com/sword4869/mytree.git#egg=mytree` 则不会。

```
>>> import mytree
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'mytree'
```

其实是因为前者安装到`\envs\nerf\Lib\site-packages`中少了东西。

```
\envs\nerf\Lib\site-packages`
├── mytree                      # 这个文件夹因为没有 __init__.py 而没有
│   ├── __init__.py
│   ├── tree.py
│   └── __pycache__
├── mytree-0.0.1.dist-info
│   ├── INSTALLER
│   ├── LICENSE
│   ├── METADATA
│   ├── RECORD
│   ├── REQUESTED
│   ├── WHEEL
│   ├── direct_url.json
│   ├── entry_points.txt
│   └── top_level.txt
```

有了`__init__.py` 后，这下不用 `-e` 和 用 `-e` 都行。

### build


```bash
# 安装build包
pip install build
```


If you run `build` without `--wheel` or `--sdist`, it will build both files for you.

```bash
python -m build .
```

#### build a wheel
二进制文件

```bash
# 使用build包打包
python -m build --wheel .
```


#### build source distribution (sdist)

- 包含源码
- 是一个压缩包`*.tar.gz`


```bash
python -m build --sdist . 
```

默认包含在 sdist 中的文件:
-   all Python source files implied by the `py_modules` and `packages`
    
-   all C source files mentioned in the `ext_modules` or `libraries`
    
-   scripts specified by the `scripts`
    
-   all files specified by the `package_data` and `data_files` 
    
-   the file specified by the `license_file` option in `setup.cfg` (setuptools 40.8.0+), all files specified by the `license_files` option in `setup.cfg` (setuptools 42.0.0+)
    
-   all files matching the pattern `test/test*.py`
    
-   `setup.py` (or whatever you called your setup script)
-   `setup.cfg`
-   `pyproject.toml` (setuptools 43.0.0+)
    
-   `README`, `README.txt`, `README.rst` (Python 3.7+ or setuptools 0.6.27+), `README.md` (setuptools 36.4.0+)
    
-   `MANIFEST.in`


Default files can even be removed from the sdist with `MANIFEST.in`.

> `MANIFEST.in`


您可能会发现自己想要在源发行版中包含额外的文件，例如`contributors`文件、`docs/`文件夹，或用于测试目的的数据文件目录`data/`。

方法是：添加和删除文件是通过在项目根目录下写入`MANIFEST.in`文件来完成的。



| 命令（文件必须以项目根目录为相对路径给出, 用`/`分隔） | 描述 |
| --- | --- |
| `include pat1 pat2 ...` | 添加文件 |
| `exclude pat1 pat2 ...` | 删除文件 |
| `recursive-include dir-pattern pat1 pat2 ...` | 添加`dir-pattern`目录下的文件 |
| `recursive-exclude dir-pattern pat1 pat2 ...` | 删除`dir-pattern`目录下的文件 |
| `global-include pat1 pat2 ...` | 添加所有在源代码树中任何位置并且与任何列出的模式匹配的文件 |
| `global-exclude pat1 pat2 ...` | 删除所有在源代码树中任何位置并且与任何列出的模式匹配的文件 |
| `graft dir-pattern` | 添加所有在`dir-pattern`目录下的文件 |
| `prune dir-pattern` | 删除所有在`dir-pattern`目录下的文件 |

`graft example*` will include a directory named `examples` in the project root but will not include `docs/examples/`.


---

https://docs.python.org/3/distutils/setupscript.html#distutils-installing-scripts

https://setuptools.pypa.io/en/latest/userguide/datafiles.html
