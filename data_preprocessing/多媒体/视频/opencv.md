- [1. video2squence](#1-video2squence)
- [2. squence2video](#2-squence2video)


---
## 1. video2squence
```python
import os 
import cv2

def video2sequence(video_path):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.split(videofolder)[-1]
    print(1)

    vidcap = cv2.VideoCapture(video_path)
    print(2)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(videofolder, f"{video_name}_frame_{count:04d}.png")
        cv2.imwrite(imagepath, image)
        success, image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
        print(f"{count} video frames are stored in {videofolder}")
    return imagepath_list
```

## 2. squence2video

写入视频，如同写入图片一样，都需要uint8格式的数据

```python
import cv2

def squence2video(imgs, savename, fps=25):
    '''save a sequence of images to a video
    imgs: [N, H, W, C], RGB, 0~255
    fps: frames per second
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    h, w = imgs.shape[1:3]  
    out = cv2.VideoWriter(savename, fourcc, fps, (w, h))
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    out.release()
    print('save video to', savename)
```

- `MPEG`: `.avi`, `.mp4`
- `mp4v`: `.avi`, `.mp4`。注意`MP4V`不行。
- `MJPG`: `.avi`可以，`.mp4`不行
- `XVID`: `.avi`可以，`.mp4`不行