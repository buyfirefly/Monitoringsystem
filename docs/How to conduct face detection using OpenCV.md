## 输入图像路径
```python
import cv2
imagePath = 'input_image.jpg'
```
## 读取图像
```python
img = cv2.imread(imagePath)
```
- `imread()` 读取图像函数
	- 从指定文件路径加载图像
	- 以`Numpy` 数组的形式返回
- 打印这个数组的维度
```python
img.shape
```

```output
(4000, 2667, 3)
```
- 数组的值代表图片高度、宽度和通道(channels)
	- 通道（BGR）
## 将图像转换为灰度  
```python
grav_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

- 函数 `cv2.cvtColor()` 接受两个参数：
	- 第一个参数是要转换的图像
	- 第二个参数是颜色转换的代码
- `cv2.COLOR_BGR2GRAY` 表示将图像从 BGR 色彩空间（OpenCV 默认使用的）转换为灰度图像。

```python
gray_image.shape
```

```python
(4000, 2667)
```

## 加载分类器
- 加载内置于 OpenCV 的预训练的 Hear 级联分类器
```python
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
```
- 这段代码使用 `haarcascade_frontalface_default.xml` 文件初始化了一个名为 `face_classifier` 的 `CascadeClassifier` 对象。`CascadeClassifier` 是 OpenCV 库中的一个类，用于进行目标检测。而 `haarcascade_frontalface_default.xml` 是一个预训练的分类器文件，用来检测图像或视频流中的正面人脸。

## 执行人脸检测
```python
face = face_classifer.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5, minSize = (40, 40))
```
`detectMultiScale` 函数接受以下参数：
- **gray_image**：这是转换为灰度的图像，是我们想要在其中检测人脸的图像版本。
- **scaleFactor**：这个参数用于补偿图像中人脸可能具有不同尺寸的情况。值为 `1.1` 表示算法每次图像尺寸缩小 10% 来寻找不同大小的人脸。    
- **minNeighbors**：该参数指定每一个候选矩形保留前需要有多少邻近矩形。值越大，检测到的人脸越少但准确率更高。
- **minSize**：指定检测人脸的最小尺寸，小于该尺寸的人脸将被忽略。

## 绘制边界框
```python
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
```
- `face` 变量检测到人脸的 x 轴和 y 轴坐标，以及人脸的宽度和高度
- 参数 `0,255,0` 表示边界框的颜色，为绿色，而 `4` 表示其厚度

## 显示图像
- 把 BGR 转化 RGB 格式
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
- 用Matplotlib 库来显示图像
```python
import matplotlib.pyplot as plt

plt.figure(figsize = (20, 10))
plt.imshow(img_rgb)
plt.axis('off')
```
- 这段代码导入了`matplotlib.pyplot`模块，
	- `plt.figure()`函数创建了一个大小为20x10的新图形
	- `plt.imshow()`函数显示图像`img_rgb`
	- `plt.axis('off')`函数关闭坐标轴

- `matplotlib.pyplot`模块是一组函数的集合，提供了创建图表和图形的便捷接口
	- `plt.figure()`函数创建一个指定大小的新图形
	- `plt.imshow()`函数在图形上显示图像。`plt.axis('off')`函数关闭坐标轴，坐标轴是指示图表比例的一组线条

```python
import cv2
import matplotlib.pyplot as plt

def face_identification(imagePath):
    img = cv2.imread(imagePath)
    grav_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face = face_classifier.detectMultiScale(grav_image, scaleFactor = 1.1, minNeighbors = 5, minSize = (40, 40))
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize = (20, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def main():
    imagePath = '/Users/mikey233/Desktop/IMG_8730.PNG'
    face_identification(imagePath)

if __name__ == "__main__":
    main()

```
