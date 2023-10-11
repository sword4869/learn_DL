[toc]

---

[[toc]]

---

- [1. wave](#1-wave)
  - [1.2. wave](#12-wave)
    - [1.2.1. 读取](#121-读取)
    - [1.2.2. 写入](#122-写入)
  - [1.3. 滤波](#13-滤波)

---

# 1. wave

## 1.2. wave

### 1.2.1. 读取
```python
"""
这里的frame都是采样
"""

import wave

audio_path = "test/test-something.wav"
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
n_bytes = len(data)  # 多少个字节
print(n_bytes)  # 133120

# 时长
n_seconds = n_samples / sample_rate
print(n_seconds)

wf.close()
```

```python
'''
输入到numpy中
'''
import numpy as np
# all data from n channels as a 1-dimensional array
signal_array = np.frombuffer(data, dtype=np.int16)  # 采样宽度保持一致
print(len(signal_array))    # 音频总采样数 n_samples * 通道数 n_channels
```
### 1.2.2. 写入

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

## 1.3. 滤波

```python
import math
import wave

import numpy as np


class LowPassFilter:
    def __init__(self) -> None:
        self.cutOffFrequency = 400.0

    # from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    def running_mean(self, x, windowSize):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

    # from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
    def interpret_wav(self, raw_bytes, n_samples, n_channels, sample_width, interleaved=True):

        if sample_width == 1:
            dtype = np.uint8  # unsigned char
        elif sample_width == 2:
            dtype = np.int16  # signed 2-byte short
        else:
            raise ValueError("Only supports 8 and 16 bit audio formats.")

        signal_array = np.frombuffer(raw_bytes, dtype=dtype)

        if interleaved:
            # signal_array are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
            signal_array.shape = (n_samples, n_channels)
            signal_array = signal_array.T
        else:
            # signal_array are not interleaved. All samples from channel M occur before all samples from channel M-1
            signal_array.shape = (n_channels, n_samples)

        return signal_array

    def filter_bytes(self, raw_bytes, sample_rate, sample_width, n_channels, n_samples):
        signal_array = self.interpret_wav(raw_bytes, n_samples, n_channels, sample_width, True)

        # get window size
        # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        freqRatio = self.cutOffFrequency / sample_rate
        N = int(math.sqrt(0.196196 + freqRatio**2) / freqRatio)

        # Use moviung average (only on first channel)
        filtered = self.running_mean(signal_array[0], N).astype(signal_array.dtype)
        alter_bytes = filtered.tobytes()

        return alter_bytes

    def filter_file(self, input_path, output_path):
        wf = wave.open(input_path, "rb")
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()
        n_samples = wf.getnframes()
        data = wf.readframes(n_samples)
        wf.close()

        alter_bytes = self.filter_bytes(data, sample_rate, sample_width, n_channels, n_samples)
        self.write_to_file(output_path, alter_bytes, sample_rate, sample_width, n_channels)

    def write_to_file(self, output_path, data, sample_rate, sample_width, n_channels):
        wf = wave.open(output_path, "wb")
        wf.setnchannels(n_channels) # 通道数
        wf.setsampwidth(sample_width) # 采样格式
        wf.setframerate(sample_rate) # 采样频率
        wf.writeframes(data)
        wf.close()


def main():
    input_path = "test/test-something.wav"
    output_path = "test/filtered.wav"


    lowPassFilter = LowPassFilter()

    # 过滤文件
    lowPassFilter.filter_file(input_path, output_path)

    # 过滤字节
    wf = wave.open(input_path, "rb")
    sample_rate = wf.getframerate()
    sample_width = wf.getsampwidth()
    n_channels = wf.getnchannels()
    n_samples = wf.getnframes()
    data = wf.readframes(n_samples)
    wf.close()
    lowPassFilter.filter_bytes(output_path, data, sample_rate, sample_width, n_channels, n_samples)
    print("* done")


if __name__ == "__main__":
    main()
```