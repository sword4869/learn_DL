You are responsible for calling `model.eval()` and `model.train()` if your model relies on modules such as `torch.nn.Dropout` and `torch.nn.BatchNorm2d` that may behave differently depending on training mode, for example, to avoid updating your BatchNorm running **statistics on validation data**.
## Welford算法小记


[Welford算法小记](https://zhuanlan.zhihu.com/p/4084747100)

```python
import numpy as np
x_arr = np.array([1, 2, 3, 4, 5]).astype(np.float32)

def naive_update(sum, sum_square, new_val):
    sum = sum + new_val
    sum_square = sum_square + new_val * new_val
    return (sum, sum_square)

# 只需要保存 sum, sum_square
naive_sum = 0
naive_sum_square = 0
for i in range(len(x_arr)):
    new_val = x_arr[i]
    naive_sum, naive_sum_square = naive_update(naive_sum, naive_sum_square, new_val)
    naive_mean = naive_sum / (i + 1)
    naive_var = naive_sum_square/ (i + 1) - naive_mean*naive_mean
    print(
        f"naive_mean: {naive_mean:3.2f}", 
        f"naive_var: {naive_var:3.2f}",
        f"naive_sum: {naive_sum:3.2f}",
        f"naive_sum_square: {naive_sum_square:3.2f}",
    )

def welford_update(count, mean, M2, new_val):
    count += 1
    delta = new_val - mean
    mean += delta / count
    delta2 = new_val - mean
    M2 += delta * delta2
    return (count, mean, M2)

# 保存 welford_count， welford_mean，welford_m2
welford_mean = 0
welford_m2 = 0
welford_count = 0
for i in range(len(x_arr)):
    new_val = x_arr[i]
    welford_count, welford_mean, welford_m2 = welford_update(welford_count, welford_mean, welford_m2, new_val)
    welford_var = welford_m2 / welford_count
    print(
        f"welford_count: {welford_count:3.2f}", 
        f"welford_mean: {welford_mean:3.2f}",
        f"welford_var: {welford_var:3.2f}",
        f"welford_m2: {welford_m2:3.2f}",
    )
```

## BN
一个batch一般都只有1-2张图片，不建议使用 BN。因为BN一般是16张图片以上一起跑。