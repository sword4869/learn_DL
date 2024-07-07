好处就是读取、显示、保存，uint8 或者 float 都行。

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image -> nparry
img = Image.open(r'D:\git\learn_python\images\8a85432a391c379d6006ddf59e77df7fd138f2a80e115058abbd06431d727d30.png')
array = np.array(img)

plt.imsave('f1f.png', array)
plt.imshow(array)
plt.show()
```


显示单通道图有一个bug，如果全是1或者全是0，那么会显示黑色。存在多个值时，1才是纯白，0是纯黑。
```python
plt.imshow(img[..., 3:4], cmap='gray')
```