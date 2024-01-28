# Installation

```bash
pip install ffmpy
```

# Quick example: 格式转化

```python
import ffmpy
ff = ffmpy.FFmpeg(
    inputs={'input.mp4': None},
    outputs={'output.avi': None}
)
ff.run()
```
This will take input.mp4 file in the current directory as the input, change the video container from MP4 to AVI without changing any other video parameters and create a new output file output.avi in the current directory.