- [1. wave](#1-wave)
  - [1.1. wave](#11-wave)
    - [1.1.1. 读取](#111-读取)
    - [1.1.2. 写入](#112-写入)
  - [1.2. wavefile](#12-wavefile)


---

# 1. wave

## 1.1. wave

### 1.1.1. 读取
```python
"""
这里的frame都是采样
"""

import wave

audio_path = r"C:\FFOutput\1.wav"
wf = wave.open(audio_path, "rb")

# 通道数
n_channels = wf.getnchannels()  # 1

# 采样宽度
sample_width = wf.getsampwidth()  # 2

# 采样频率
sample_rate = wf.getframerate()  # 16000

# 音频总共有多少采样
n_samples = wf.getnframes()  # 66560

# 一起返回上述信息
params = wf.getparams() 
print(params)   # _wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=66560, comptype='NONE', compname='not compressed')

# 读取全部字节
data = wf.readframes(n_samples)
n_bytes = len(data)  # 多少个字节 Bytes
print(n_bytes)  # 133120

# 时长
n_seconds = n_samples / sample_rate
print(n_seconds)

wf.close()
'''
# mono wav
_wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=160085, comptype='NONE', compname='not compressed')

# stereo wav
_wave_params(nchannels=2, sampwidth=2, framerate=44100, nframes=43823, comptype='NONE', compname='not compressed')
'''
```

```python
'''
输入到numpy中
'''
import numpy as np
import wave

audio_path = r"C:\FFOutput\1.wav"
wf = wave.open(audio_path, "rb")
n_samples = wf.getnframes()
data = wf.readframes(n_samples)
print(wf.getparams())

# all data from n channels as a 1-dimensional array
# 音频总采样数 * 通道数 
# n_samples * n_channels
signal_array = np.frombuffer(data, dtype=np.int16)  # 采样宽度保持一致
print(signal_array.shape)
'''
# mono wav
_wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=160085, comptype='NONE', compname='not compressed')
(160085,)
# 160085 = 160085 * 1 

# stereo wav
_wave_params(nchannels=2, sampwidth=2, framerate=44100, nframes=43823, comptype='NONE', compname='not compressed')
(87646,)
# 87646 = 43823 * 2
'''
```
### 1.1.2. 写入

```python
import wave

audio_path = 'test/test-something.wav'
wf = wave.open(audio_path, 'wb')

wf.setnchannels(1) # 通道数
wf.setsampwidth(2) # 采样格式
wf.setframerate(16000) # 采样频率

wf.writeframes(b''.join(data))   # 写入字节

wf.close()
```

## 1.2. wavefile

```python
from scipy.io import wavfile

audio_path = r"C:\FFOutput\audio.wav"
audio_sample_rate, audio = wavfile.read(audio_path)
n_channels = audio.ndim
print(n_channels)
print(audio_sample_rate)
print(audio.shape)      # ()
'''
# mono wav
1
(160085,)
16000

# stereo wav
2
(43823, 2)
44100
'''
```