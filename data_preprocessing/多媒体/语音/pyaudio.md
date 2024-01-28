[toc] 

---

[[toc]]

---

- [1. pyaudio](#1-pyaudio)
  - [1.1. Intro](#11-intro)
  - [1.2. pyaudio](#12-pyaudio)

---

# 1. pyaudio
## 1.1. Intro


> Installation

```bash
# linux: before pip install
# sudo apt-get install portaudio19-dev
pip install pyaudio
pip install wave
```

## 1.2. pyaudio

```python
import time
import wave

import pyaudio


class MyAudio:
    # 定义音频属性
    frames_per_buffer = 1024
    pyaudio_format = pyaudio.paInt16
    n_channels = 1
    sample_rate = 16000
    audio_path = "output.wav"  
    
    def __init__(self):
        pass
        
    def record_start(self):
        ### 
        # 不定时长录音之启动
        ###

        # 创建PyAudio对象
        self.p = pyaudio.PyAudio()

        # 创建wave对象        
        self.wf = wave.open(self.audio_path, 'wb')
        self.wf.setnchannels(self.n_channels)
        self.wf.setsampwidth(self.p.get_sample_size(self.pyaudio_format))
        self.wf.setframerate(self.sample_rate)


        def callback(in_data, frame_count, time_info, status):
            self.wf.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        # 打开数据流
        self.stream = self.p.open(format=self.pyaudio_format,
                        channels=self.n_channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.frames_per_buffer,
                        stream_callback=callback)

        print("* recording")

        # 开始录音
        # start_stream()之后stream会开始调用callback函数
        self.stream.start_stream()


    def record_stop(self):   
        ### 
        # 不定时长录音之停止
        ###
        #      
        self.stream.stop_stream()   # 停止数据流
        self.stream.close()
        self.p.terminate()          # 关闭 PyAudio
        self.wf.close()             # 关闭 wave

        print("* done recording")

    def record_seconds(self, seconds):
        ### 
        # 定时长录音
        ###

        # 创建PyAudio对象
        p = pyaudio.PyAudio()

        # 创建wave对象        
        wf = wave.open(self.audio_path, 'wb')
        wf.setnchannels(self.n_channels)
        wf.setsampwidth(p.get_sample_size(self.pyaudio_format))
        wf.setframerate(self.sample_rate)

        # 打开数据流
        stream = p.open(format=self.pyaudio_format,
                        channels=self.n_channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.frames_per_buffer,
                    )

        # 一次全读完
        data = stream.read(self.sample_rate * seconds)
        wf.writeframes(data)

        print(f"* record done {seconds} seconds")

        stream.stop_stream()   # 停止数据流
        stream.close()
        p.terminate()          # 关闭 PyAudio
        wf.close()             # 关闭 wave

    def record_play(self):
        ### 
        # 播放录音
        ###

        wf=wave.open(self.audio_path, 'rb')

        p=pyaudio.PyAudio()
        stream=p.open(
            format=p.get_format_from_width(width=wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        print('* play...')
        

        #### 写入就是播放，都是阻塞式
        ## 方法一：逐步写入
        # while True:
        #     data=wf.readframes(self.frames_per_buffer)
        #     if data==b"":   # 空字节退出
        #         break
        #     stream.write(data)

        ## 方法二：直接全部写入
        stream.write(wf.readframes(wf.getnframes()))
        
        stream.stop_stream()   # 停止数据流
        stream.close()
        p.terminate()          # 关闭 PyAudio
        wf.close()             # 关闭 wave

        print('* done play')

if __name__ == '__main__':
    myAudio = MyAudio()
    
    # 5秒录音
    # myAudio.record_seconds(5)   
    
    # 不定时录音
    myAudio.record_start()
    time.sleep(3)
    myAudio.record_stop()
    
    # 播放录音
    myAudio.record_play()
```