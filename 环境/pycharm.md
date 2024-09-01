https://blog.csdn.net/qq_67822268/article/details/136546887

## 下载补丁

在[JETBRA.IN CHECKER | IPFS](https://3.jetbra.in/)中找到一个能用的网站。

比如，hardbin，下载补丁

![image-20240709170748550](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407091708894.png)

## 将补丁放到任意非中文路径下

将补丁文件（注意是整个文件）移动到**与Pycharm同一目录下**

	$ sudo mv pycharm-2024.2.0.1  /usr/local/pycharm
	$ sudo mv jetbra  /usr/local
	
	lab@labU:/usr/local/pycharm$ ls
	bin         help                   lib      plugins
	build.txt   Install-Linux-tar.txt  license  product-info.json
	debug-eggs  jbr                    modules
	
	lab@labU:/usr/local/jetbra$ ls
	config-jetbrains  plugins-jetbrains  readme.txt  sha1sum.txt
	ja-netfilter.jar  README.pdf         scripts     vmoptions

进入/usr/local/pycharm/bin目录下，修改配置文件pycharm64.vmoptions

​	-javaagent: 开头，后面跟着补丁的绝对路径（可根据你实际的位置进行修改）,注意路径一定要填写正确，且不能包含中文，否则会导致 Pycharm 无法启动

​	**切勿**在修改配置文件时加空格，会**报错**找不到或无法加载主类

```
-javaagent:/usr/local/jetbra/ja-netfilter.jar
--add-opens=java.base/jdk.internal.org.objectweb.asm=ALL-UNNAMED
--add-opens=java.base/jdk.internal.org.objectweb.asm.tree=ALL-UNNAMED
```

使用命令pycharm.sh启动pycharm

```
lab@labU:/usr/local/pycharm/bin$ ./pycharm.sh

  ============================================================================  

    ja-netfilter 2022.2.0

    A javaagent framework :)

    https://github.com/ja-netfilter/ja-netfilter
```



![image-20240709170634910](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407091706958.png)

## 复制激活码

![image-20240709170541849](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407091705928.png)

## 

## 打开idea，填入激活码

如果在之前就打开了idea，那么显示 key is invalid。

本激活教程原理就是 阻止软件网络请求验证是否过期，其中的 有效期 、激活来源 均在配置文件（config/mymap.conf）中配置，补丁来源于jetbra.in。

我们可以在About查看激活有效期，有效期是多少无所谓，它是备用许可证，不会过期。



![image-20240709171307116](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407091713206.png)

## conda卡住

Pycharm打开加载环境时卡在Scanning installed packages，然后让你输入yN，但是不能输入。

这个问题其实是conda的问题。

去控制台，激活对应的conda环境，然后conda list

```
(3d) C:\Users\lab>conda list

# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

    Traceback (most recent call last):
      File "D:\Applications\miniconda\lib\site-packages\conda\exc
```

搞了半天，似乎只能重创个虚拟环境？