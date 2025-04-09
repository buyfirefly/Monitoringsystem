import numpy as np
import cv2
import matplotlib.pyplot as plt

# 生成一个渐变图像（高=256，宽=256，3通道彩色）
img = np.zeros((256, 256, 3), dtype=np.uint8)

for i in range(256):
    img[:, i, 0] = i        # Blue 渐变
    img[i, :, 1] = i        # Green 渐变
    img[i, i, 2] = 255      # 红色对角线

# 使用 OpenCV 显示图像
cv2.imshow('OpenCV Display', img)
cv2.waitKey(0)   # 等待按键
cv2.destroyAllWindows()

# 使用 matplotlib 显示图像（注意颜色通道顺序）
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Matplotlib Display')
plt.axis('off')
plt.show()

