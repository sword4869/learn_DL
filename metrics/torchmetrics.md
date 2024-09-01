

```bash
pip install torchmetrics[image] 

pip install lpips
```



```python
import torch
import numpy as np
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import peak_signal_noise_ratio


# 实验复现
_ = torch.manual_seed(123)
np.random.seed(123)

class Metric:
    def __init(self, device='cpu'):
        self.mean_absolute_error = MeanAbsoluteError().to(device)
        self.mean_squared_error = MeanSquaredError().to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
        self.psnr = PeakSignalNoiseRatio().to(device)
        
    def step_each_batch(self, preds, imgs):
        """
        # float32 要么在 np.astype() , 要么在 tensor.dtype 处
        imgs1 = np.random.rand(10, 100, 100, 3).astype(np.float32)
        imgs1 = torch.tensor(imgs1, device='cuda:0').permute(0, 3, 1, 2)
        imgs2 = np.random.rand(10, 100, 100, 3)
        imgs2 = torch.tensor(imgs2, dtype=torch.float32, device='cuda:0').permute(0, 3, 1, 2)
        """
        # step batch
        self.mean_absolute_error(preds, target)
        self.mean_squared_error(preds, target)
        self.lpips(preds, target)
        self.psnr(preds, target)
        
    def compute_all_batch(self):
        # total accuracy over all training batches
        mae_metric = self.mean_absolute_error.compute()
        mse_metric = self.mean_squared_error.compute()
        lpips_metric = self.lpips.compute()
        psnr_metric = self.psnr.compute()
            
		metrics = {
            'L1': mae_metric,
            'mse':mse_metric,
            'lpips':lpips_metric,
            'psnr':psnr_metric
        }
        
        # Reset metric states after each epoch
        self.mean_absolute_error.reset()
        self.mean_squared_error.reset()
        self.lpips.reset()        
        self.psnr.reset()        
        
        return metrics
```

