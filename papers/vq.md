vector quantization (VQ) 

## numpy example

The basic operation in VQ finds the closest point in a set of points, called `codes` in VQ jargon, to a given point, called the `observation`. Finding the closest point requires calculating the distance between `observation` and each of the `codes`. The shortest distance provides the best match. 

- In the very simple, two-dimensional case shown below, the values in `observation` describe the weight and height of an athlete to be classified. 
- The `codes` represent different classes of athletes.
- In this example, `codes[0]` is the closest class indicating that the athlete is likely a basketball player.

```python
>>> from numpy import array, argmin, sqrt, sum
>>> observation = array([111.0, 188.0])
>>> codes = array([[102.0, 203.0],
...                [132.0, 193.0],
...                [45.0, 155.0],
...                [57.0, 173.0]])
>>> diff = codes - observation    # the broadcast happens here
>>> dist = sqrt(sum(diff**2,axis=-1))
>>> argmin(dist)
0
```

In this example, the `observation` array is stretched to match the shape of the `codes` array:

```
Observation      (1d array):      2
Codes            (2d array):  4 x 2
Diff             (2d array):  4 x 2
```

![图 1](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407062012559.png)