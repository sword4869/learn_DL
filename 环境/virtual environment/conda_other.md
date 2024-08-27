[toc]

# conda还可以安装什么

## packages that conda can install

```python
conda install git

# 系统python没有pip时，我们可以这样给系统python安装pip
conda install pip
```
## Microsoft Visual C++ 14.0 

> 直接Visual Studio 吧

以为百度云的就行: <https://pan.baidu.com/s/1fQdjG3cgROgrV4J_k2fQiw?pwd=4g4s>

但有些代码会去找`cl.exe`，比如, `paths = sorted(glob.glob(f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"), reverse=True)`, `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64\cl.exe`

这没办法。