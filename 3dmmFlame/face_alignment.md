```python
class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'

bbox, bbox_type = face_detector.run(image)
if len(bbox) < 4:
    print('no face detected! run original image')
    left = 0; right = h-1; top=0; bottom=w-1
else:
    left = bbox[0]; right=bbox[2]
    top = bbox[1]; bottom=bbox[3]
```

# face_alignment

## 2维

```python
import cv2
import face_alignment
from skimage import io
from skimage.transform import resize

def cv_draw_landmark(img_input, pts, show_text=False, color=(0, 0, 255)):
    '''
    img_input: 0-255, uint8, rgb, [h, w, 3]
    pts: [n, 2]
    '''
    img = img_input.copy()
    size = int(img.shape[0] // 256)
    n = pts.shape[0]
    for i in range(n):
        cv2.circle(img, (int(round(pts[i, 0])), int(round(pts[i, 1]))), size, color, -1)
        if show_text:
            cv2.putText(img, str(i), (int(round(pts[i, 0])-3), int(round(pts[i, 1])-3)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0,255,0), 1)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    return img

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

input = io.imread(r'D:\DataSet\nersemble\nersemble_018_EMO-1-shout+laugh\018\sequences\EMO-1-shout+laugh\frame_00000\images-2x-73fps\cam_220700191.png')
resize_input = resize(input, (256, input.shape[1] * 256 // input.shape[0]))
resize_input = (resize_input*255).astype('uint8')   

output = resize_input
# 需要 uint8, rgb, [h, w, 3]
preds = fa.get_landmarks(output)
# print(preds)    # 列表，几个图片。没有检测到人脸时，为None
# print(preds[0].shape)   # (68, 2), 68个点，2维

output = cv_draw_landmark(output[..., ::-1], preds[0])
```
## 3维

```python
# 需要 uint8, rgb, [h, w, 3]
preds = fa.get_landmarks(output)
# print(preds)    # 列表，几个图片。没有检测到人脸时，为None
# print(preds[0].shape)   # (68, 3), 68个点，3维
```

## Running on CPU/GPU

```python
import torch
import face_alignment

# cuda for CUDA, mps for Apple M1/2 GPUs.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

# running using lower precision
fa = fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, dtype=torch.bfloat16, device='cuda')
```
