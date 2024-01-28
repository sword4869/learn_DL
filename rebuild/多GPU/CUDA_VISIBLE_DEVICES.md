- 只在当前Terminal生效（即如果打开一个新的Terminal 窗口，定位到当前目录，打印PYTHONPATH 是没有刚才加入的路径的）
    - linux
      ```bash
      export CUDA_VISIBLE_DEVICES=0
      export CUDA_VISIBLE_DEVICES="0,1"
      ```
      然后执行
      ```bash
      ## script or interactive python
      python xxx.py
      python
      ```
    - windows

      注意 `set` 环境变量是在cmd下。
      ```bash
      # set CUDA_VISIBLE_DEVICES="0" 是错的
      
      set CUDA_VISIBLE_DEVICES=0
      set CUDA_VISIBLE_DEVICES=0,1
      ```
- 只对这一次命令起效果
    - 只有linux
      ```bash
      CUDA_VISIBLE_DEVICES=0 python xxx.py
      CUDA_VISIBLE_DEVICES=0 python
      ```
- 只对此脚本内有效果
    win/linux通用
    ```python
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    import torch
    print(torch.cuda.device_count())
    # 1
    ```