## np.random.choice()
<https://blog.csdn.net/ImwaterP/article/details/96282230>

## np.newaxis 
<https://zhuanlan.zhihu.com/p/356601576>
- 是切片
- `y[:, np.newaxis, :] == y[...,  np.newaxis, :]`, [4,1,3]
- `y[:, np.newaxis, :].shape`, [4, 1 ,3]; `y[:, np.newaxis, :].shape`, [4, 3, 1]



## np.concatenate

The arrays must have the same shape, except in the dimension corresponding to axis (the first, by default).
```python
# numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")

# a: [2, 2], b: [1, 2]
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# [2+1, 2] = [3, 2]
np.concatenate([a, b])
# array([[1, 2],
#        [3, 4],
#        [5, 6]])

# [2, 2+1] = [2, 3]
np.concatenate((a, b.T), axis=1)
# array([[1, 2, 5],
#        [3, 4, 6]])

# If `axis` is `None`, arrays are flattened before use. 
np.concatenate((a, b), axis=None)
# array([1, 2, 3, 4, 5, 6])
```