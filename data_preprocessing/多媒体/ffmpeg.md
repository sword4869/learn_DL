[toc]

<http://ffmpeg.org/download.html>

<https://itsfoss.com/ffmpeg/>

```bash
conda install ffmpeg
# 而不是 pip install ffmpeg, 这个下的是假库
```

## 视频

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

### 转换视频格式
```bash
# 转换格式
ffmpeg -i video_input.mp4 video_output.avi 
ffmpeg -i audio_input.mp3 audio_output.ogg 

# specify more output files:
ffmpeg -i audio_input.wav audio_output_1.mp3 audio_output_2.ogg
```
### 限定视频fps
```bash
# -r 20         fps
ffmpeg -i video_input.mp4 -r 20 video_output.mp4
```


### 重复写入不报错
```bash
# -y          override
ffmpeg -i video_input.mp4 -y video_output.mp4
```

## 提取声音


### Extract Audio from Audio

```bash
# -ar 16000     audio read 采样频率
# -ac 1         audio count 通道数
# -ab 256k      audio bitrate 比特率
ffmpeg -i 001.mp3 -ar 16000 -ac 1 001.wav 
```

### Extract Audio from Video
```bash
# -vn                 video no

# 输出是视频格式，则是没有画面的视频
ffmpeg -i video.mp4 -vn video_no_audio.mp4

# 输出是音频格式，则可以不用
ffmpeg -i video.mp4 audio.mp3
```
综合命令：
```bash
ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.mp3
```

### 静音

```bash
# -an                  audio no
ffmpeg -i input.mp4 -an output.mp4
```

### 音量增强


```bash
# Audio Volume Multiplying
# The first command amplifies the volume 1.5 times. The second command makes the audio 1/4 (0.25) times quieter.

ffmpeg -i input.wav -af "volume=1.5" output.wav
ffmpeg -i input.ogg -af "volume=0.75" output.ogg
```

## 从视频中提取图片

```bash
#  -f image2: 不写也行。告诉ffmpeg你想要的输出是图片而不是另一个视频文件或音频文件
# "%05d.png": 从 00001 开始
ffmpeg -i "video.mov" -f image2 "%05d.png"
```
```bash
# -i 输入文件
# -vf fps=25 输出图片的帧率25，默认采用输入源的帧率
# -q:v 1 最高质量输出
os.system(f'ffmpeg -i {video_file} -vf fps=25 -q:v 1 {self.source}/source/%05d.png')
```

## 合成图片到视频

```bash
# -y: 已经存在，直接覆盖文件。
# -framerate 25: 输出视频的帧率为25
# -pattern_type glob: 这个选项与-i选项一起使用，指定了输入文件的模式类型为glob，这意味着可以使用通配符（如*）来匹配一组文件。
# -i '{actor}/video/*.jpg': -i选项指定了输入文件。在这里，它使用了glob模式来匹配所有在{actor}/video/目录下的.jpg文件。
# -c:v libx264: 视频编码器为libx264，这是一种非常流行的开源H.264/AVC编码器，用于生成MP4格式的视频。
os.system(f'ffmpeg -y -framerate 25 -pattern_type glob -i \'{actor}/video/*.jpg\' -c:v libx264 {actor}.mp4')
```

## 由图片生成gif
```bash
# -i                  图片格式, 01.png,02.png
ffmpeg -i "%01d.png" -r 20 name.gif
```

## crop

```bash
# -vf crop=in_w-200:in_h-200 
# -vf crop=640:480:0:0

ffmpeg -i out.flv -vf crop=in_w-200:in_h-200 out.mp4
```

## 音视频时间裁剪

```bash
# -ss HH:MM:SS.MILLISECONDS    start second 开始时间
# -t 10                        裁剪的时间大小，这里裁剪10s
ffmpeg -i out.flv -ss 00:00:00 -t 10 out.ts
```
```bash
# -ss HH:MM:SS.MILLISECONDS    start second 开始时间
# -to HH:MM:SS.MILLISECONDS    结束时间 
ffmpeg -i out.flv -ss 40 -to 70 out.ts
```


## 格式

```bash
ffmpeg -i out.flv -f s16le out.pcm
ffmpeg -i input.mp4 -c:v rawvideo -pix_fmt yuv420p out.yuv
ffmpeg -i out.flv -c:v copy -c:a copy out.mp4
```
`s16le`: s代表有符号 16表示每一个数值用16位表示 le表示为little end 小头存储

`-c:v` 等于`-vcodec`
  ```
  copy
  rawvideo
  libx264 
  ```

`-c:a` 等于 `-acodec`
  ```
  copy
  libmp3lame (mp3)
  pcm_s16le (wav)
  ```
`-pix_fmt` 像素格式

## 合成b站缓存音频和视频

```
ffmpeg -i video.m4s -i audio.m4s -codec copy output.mp4
```