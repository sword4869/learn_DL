[toc]
## basic

```bash
numpy           # 只包含包名.方便，但不利于控制风险。
docopt==0.6.1   # 指定版本
requests>= 2.8.1  # 自动保持版本为最新
requests==2.8.* # 大版本内最新
SomeProject>= 1.2, !=1.3, <2.0    # 保证了大版本在1.2和2之间；并且，排除了已知有问题的版本1.3
```

## Global options

- The following options have an effect on the entire pip install run
- must be specified **on their individual lines**.
- 会在安装包前都扫描完全有的options后，才安装包。所以，不论写在哪里都行，比如，一开始写在文件开头，写在文件末尾，写在安装的中间。
- 正因为全扫描完，所以如果出现冲突，就得分文件。比如，index-url不兼容（不是包安随便哪个index-url都能运行的情况），一个包指定index-url，另一个指定安装别的index-url，那么就得拆分成不同`requirements_01.txt`，一个一个文件安装。


```bash
# index-url 是主要的源（即pip配置的源）, extra-index-url 是 额外源

# index-url : 只有一个，再写一个就会代替。
-i, --index-url

--extra-index-url

--no-index

-c, --constraint

-r, --requirement

-e, --editable

-f, --find-links

--no-binary

--only-binary

--prefer-binary

--require-hashes

--pre

--trusted-host

--use-feature
```


### pytorch的例子
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117`

```
--extra-index-url https://download.pytorch.org/whl/cu117
torch 
torchvision 
torchaudio 
```

用国内源下的`torch`,`torch+gpu`都失败`torch.version.cuda = None`

### 临时源的例子
```
--index-url https://mirrors.bfsu.edu.cn/pypi/web/simple	
--trusted-host https://mirrors.bfsu.edu.cn
numpy
```

## other source
```
git+https://github.com/mattloper/chumpy.git
pip @ https://github.com/pypa/pip/archive/22.0.2.zip
./downloads/numpy-1.9.2-cp34-none-win32.whl
http://wxpython.org/Phoenix/snapshot-builds/wxPython_Phoenix-3.0.3.dev1820+49a8884-cp34-none-win_amd64.whl
```