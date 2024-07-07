## libGL.so.1
```
Traceback (most recent call last):
  File "/home/lab/project/project/temp/diff/DECA/demos/demo_reconstruct.py", line 17, in <module>
    import cv2
  File "/home/lab/miniconda3/envs/sd/lib/python3.10/site-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
  File "/home/lab/miniconda3/envs/sd/lib/python3.10/site-packages/cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
  File "/home/lab/miniconda3/envs/sd/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
- [x] 经过测试
    ```bash
    pip install opencv-python-headless
    ```
    按照requirements.txt安装可能会因为顺序出错而还报错。重新安装就行,
    ```bash
    pip uninstall opencv-python
    pip uninstall opencv-python-headless
    pip install opencv-python
    pip install opencv-python-headless
    ```
- [ ] 未测试
    ```bash
    sudo apt update
    sudo apt install libgl1-mesa-glx
    ```