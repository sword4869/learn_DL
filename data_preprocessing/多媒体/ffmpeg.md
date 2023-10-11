- [1. 视频](#1-视频)
  - [1.1. 转换视频格式](#11-转换视频格式)
  - [1.2. 限定视频fps](#12-限定视频fps)
  - [1.3. 重复写入不报错](#13-重复写入不报错)
- [2. 提取声音](#2-提取声音)
  - [2.1. Extract Audio from Audio](#21-extract-audio-from-audio)
  - [2.2. Extract Audio from Video](#22-extract-audio-from-video)
  - [2.3. 音量增强](#23-音量增强)
- [3. 从视频中提取图片](#3-从视频中提取图片)
- [4. 由图片生成gif](#4-由图片生成gif)


---

<http://ffmpeg.org/download.html>

<https://itsfoss.com/ffmpeg/>

```bash
conda install ffmpeg
# 而不是 pip install ffmpeg, 这个下的是假库
```

## 1. 视频

```bash
# 显示信息，同时也是输入文件
# -i file_name
# -hide_banner不显示ffmpeg配置信息

$ ffmpeg -i 001.mp3 -hide_banner
Input #0, mp3, from '001.mp3':
  Metadata:
    encoder         : Lavf58.29.100
  Duration: 00:00:25.66, start: 0.046042, bitrate: 32 kb/s
  Stream #0:0: Audio: mp3, 24000 Hz, mono, fltp, 32 kb/s
At least one output file must be specified
```

### 1.1. 转换视频格式
```bash
# 转换格式
ffmpeg -i video_input.mp4 video_output.avi 
ffmpeg -i audio_input.mp3 audio_output.ogg 

# specify more output files:
ffmpeg -i audio_input.wav audio_output_1.mp3 audio_output_2.ogg
```
### 1.2. 限定视频fps
```bash
# -r          fps
ffmpeg -i video_input.mp4 -r 20 video_output.mp4
```


### 1.3. 重复写入不报错
```bash
# -y          override
ffmpeg -i video_input.mp4 -r 20 video_output.mp4
```

## 2. 提取声音


### 2.1. Extract Audio from Audio

```bash
# 采样频率
-ar 44100
# 通道数
-ac 2

ffmpeg -i 001.mp3 -ar 16000 -ac 1 001.wav 
```

### 2.2. Extract Audio from Video
```bash
# -vn                 disable video
ffmpeg -i video.mp4 -vn audio.mp3
```

```bash
ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.mp3
```

### 2.3. 音量增强


```bash
# Audio Volume Multiplying
# The first command amplifies the volume 1.5 times. The second command makes the audio 1/4 (0.25) times quieter.

ffmpeg -i input.wav -af "volume=1.5" output.wav
ffmpeg -i input.ogg -af "volume=0.75" output.ogg
```

## 3. 从视频中提取图片

```bash
ffmpeg -i "video.mov" -f image2 "%05d.png"
```

## 4. 由图片生成gif
```bash
# -r                  fps
# -i                  图片格式, 01.png,02.png
ffmpeg -r 20 -i "%01d.png" "name.gif"
```
