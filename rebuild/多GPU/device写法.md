- [1. lazy load](#1-lazy-load)
- [2. single gpu number](#2-single-gpu-number)


---

## 1. lazy load
1. all tensor are specified by `args.data_device = cpu, cuda:0, cuda:1`

    这样我们可以在没有gpu上的电脑运行

    但是lazy load得单独写。

2. `args.data_device` is only responsible for loading original images while other tensors are fixed on gpu.

    无需修改就可以实现lazy load
   
    e.g. 3D GS

    ```python
    ################ load image
    # if data_device == 'cpu', it is lazy load.
    try:
        self.data_device = torch.device(data_device)
    except Exception as e:
        print(e)
        print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        self.data_device = torch.device("cuda")

    # when iterating, it is sent to gpu.
    self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
    ```
## 2. single gpu number

无需指定，`to(device=cuda:0, cuda:1)`

而是运行python时指定环境变量，`CUDA_VISIBLE_DEVICES=0 python train.py`