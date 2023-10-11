[toc]

---

[[toc]]

---

- [1. pydub](#1-pydub)
  - [1.1. Intro](#11-intro)
  - [1.2. Use](#12-use)


# 1. pydub

## 1.1. Intro

> Limitation

only `.wav` files.


> Installation

```bash
pip install pydub
conda install ffmpeg
```

## 1.2. Use
```python
import os
from pydub import AudioSegment
from pydub.playback import play
class MyAudio:
    def __init__(self) -> None:
        self.song = None
        self.in_dir = None
        self.filename_withoutDir_withoutExt = None
        self.in_format = None
        self.out_format = None
        pass

    def set_song(self, filename):
        '''
        @filename: e.g. '~/some.mp3'
        '''
        ####
        # 打开文件，视频打开也是只打开其音频
        # 找不到文件，就会直接报错 FileNotFoundError
        ####
        
        # wav,mp3,ogg,flv, 这些有专门的函数
        # self.song = AudioSegment.from_wav('some.wav')
        # self.song = AudioSegment.from_mp3('some.mp3')
        # self.song = AudioSegment.from_ogg("some.ogg")
        # self.song = AudioSegment.from_flv("some.flv")

        # mp4, wma, acc
        # wav之类的也可以
        # self.song = AudioSegment.from_file('test/1.4.2.wav', format="wav")
        # self.song = AudioSegment.from_file("some.mp4", format="mp4")
        # self.song = AudioSegment.from_file("some.wma", format="wma")
        # self.song = AudioSegment.from_file("some.aiff", format="aac")

        self.in_dir, filename_withoutDir = os.path.split(filename)
        str_array = filename_withoutDir.split('.')
        self.filename_withoutDir_withoutExt = str_array[0]
        self.in_format = str_array[1]
        self.song = AudioSegment.from_file(filename, format=self.in_format)


    def get_set_attribute(self):
        
        print(self.song.frame_rate)  
        # OUTPUT: 22050
        
        print(self.song.channels)
        # OUTPUT: 2
        
        print(self.song.sample_width )
        # OUTPUT : 2
        
        # Find Maximum amplitude
        print(self.song.max)
        # OUTPUT 32768

        # 有多长时间
        print(len(self.song))               # 247989 毫秒
        print(self.song.duration_seconds)   # 247.9891156462585 秒

        # set attribute
        wav_file_new = self.song.set_frame_rate(50)
        print(wav_file_new.frame_rate)

    def play_audio(self):
        # 直接就播放出来了，阻塞式
        play(self.song)
        print('done play')

    def save_audio(self, dir=None, format=None):
        ######
        # 保存结果
        # 可以直接进行格式转化（需要安装ffmpeg），wav保存为mp3
        ######
        # (1) 直接保存
        # self.song.export("export1.wav", format="wav")  # wav
        # self.song.export("export1.mp3", format="mp3") # mp3

        # (2) 用标签保存结果（元数据）
        # self.song.export(
        #     "export2.wav", 
        #     format="wav", 
        #     tags={'artist': 'Various artists', 'album': 'Best of 2011', 'comments': 'This album is awesome!'}
        # )

        # (3) 您可以传递一个可选的比特率参数来使用ffmpeg支持的所有语法进行输出。
        # self.song.export("export3.wav", format="wav", bitrate="192k")

        if dir is None:
            dir = self.in_dir

        if format is None:
            format = self.in_format

        self.song.export(os.path.join(dir, self.filename_withoutDir_withoutExt + '.' + format), format=format)

    def split_stereo_to_mono(self):
        b = self.song.split_to_mono()
        # 分开声道，可以选择某个声道
        print(len(b))   
        print(b[0].channels)
        b[0].export('mono.wav', format='wav')

    def set_volume(self):
        ######
        # 调整音量
        ######
        louder = self.song + 10          # 音量+10db
        slienter = self.song - 10        # 音量-10dB

        louder.export('louder.wav', format='wav')
        slienter.export('slienter.wav', format='wav')

    def effect_reverse(self):
        ######
        # 音频倒播
        ######
        reversed = self.song.reverse()  # 原来的song不变

        reversed.export('reversed.wav', format='wav')

    def effect_compose(self):
        ######
        # 切片 
        # 时间单位是 milliseconds
        ######
        beginning = self.song[:10 * 1000]   # 前10秒
        end = self.song[-5 * 1000:]         # 最后5秒
        beginning.export('beginning.wav', format='wav')
        end.export('end.wav', format='wav')


        ######
        # 直接连接合成
        ######
        # 用加法（将一个文件添加到另一个文件的末尾）
        direct_add = beginning + end
        direct_add.export('direct_add.wav', format='wav')


        # 用淡入淡出连接合成
        with_style = beginning.append(end, crossfade=1.5 * 1000) # 1.5 second crossfade
        with_style.export('with_style.wav', format='wav')


        ######
        # 拼接，重复音频
        ######
        # 直接重复
        double_beginning = beginning * 2
        double_beginning.export('double_beginning.wav', format='wav')


        # 淡入淡出，2 sec fade in, 3 sec fade out
        awesome = beginning.fade_in(2000).fade_out(3000)
        awesome.export('awesome.wav', format='wav')

def main():
    myAudio = MyAudio()
    myAudio.set_song('./some.mp3')
    myAudio.get_set_attribute()
    # myAudio.play_audio()
    myAudio.save_audio(format='wav')
    myAudio.set_volume()
    myAudio.effect_reverse()
    myAudio.effect_compose()
    myAudio.split_stereo_to_mono()

if __name__ == '__main__':
    main()
```