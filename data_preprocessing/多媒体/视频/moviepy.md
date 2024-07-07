
```
pip install moviepy
```
## 给视频加上声音

```python
from moviepy.editor import AudioFileClip

# 音频剪辑可由一个音频文件创建, 或从视频剪辑的音轨里提取出来, 或得到一个已创建的视频剪辑的音轨。
audioclip = AudioFileClip("some_audiofile.mp3")
audioclip = AudioFileClip("some_video.avi")
audioclip = VideoFileClip("some_video.avi").audio

# 合成音频剪辑
videoclip2 = videoclip.set_audio(audioclip)
```

## 保存视频
```python
from moviepy.editor import VideoFileClip

video = VideoFileClip(name)
# export as a video file or GIF
animation.write_videofile("my_animation.mp4", fps=24)
animation.write_gif("my_animation.gif", fps=24)
```
## 保存剪辑中的一帧
```python
my_clip.save_frame("frame.jpeg")        # saves the first frame
my_clip.save_frame("frame.png", t=2)    # saves the frame a t=2s
```