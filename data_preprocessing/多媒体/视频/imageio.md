```python
import imageio.v3 as iio
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

iio.imwrite("a.mp4", to8b(imgs), fps=25)
```

```python
import imageio
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
```