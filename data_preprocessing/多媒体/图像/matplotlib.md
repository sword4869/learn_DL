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