## 前提条件
```python
import cv2

face_classifier = cv2.CascadeClassifier(
i    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

## 访问网络摄像头
```python
video_capture = cv2.VideoCapture(0)
```
- 我们将参数 `0` 传递给了 VideoCapture() 函数。这告诉 OpenCV 使用设备上的默认摄像头。如果您的设备连接了多个摄像头，可以相应地更改此参数值

## 识别视频流中的人脸
```python
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces
```

## 创建实时人脸检测循环
```python
while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame
    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
```
- 这段代码创建了一个循环，通过`video_capture.read()`函数持续读取视频帧。如果帧没有成功读取，则使用`break`语句终止循环。
- 对每一帧应用`detect_bounding_box()`函数来检测人脸。使用`cv2.imshow()`函数在名为"My Face Detection Project"的窗口中显示处理后的帧。 如果用户按下"q"键，则通过`cv2.waitKey()`函数终止循环。最后，调用`video_capture.release()`函数来释放视频捕获对象，并调用`cv2.destroyAllWindows()`函数关闭所有窗口。

```python
import cv2
import matplotlib.pyplot as plt
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

def video_idy(video_capture):
    while True:
        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully
        faces = detect_bounding_box( video_frame )  # apply the function we created to the video frame
        cv2.imshow( "My Face Detection Project", video_frame )  # display the processed frame in a window named "My Face Detection Project"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows() 

def main():
    video_capture = cv2.VideoCapture(0)
    video_idy(video_capture)

if __name__ == "__main__":
    main()
```