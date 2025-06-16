# 情绪识别视频分析系统 V2.0

## 系统概述

这是一个集成了VisionWeb界面和多模态情绪识别系统的完整解决方案。用户可以通过Web界面录制或上传视频，系统会自动进行AI情绪分析并实时显示结果。

## 主要功能

### 🎥 视频录制与上传
- **实时录制**: 通过摄像头实时录制视频（自动60秒分片）
- **文件上传**: 支持上传本地视频文件
- **多格式支持**: 支持MP4、AVI、MOV等常见视频格式
- **大文件支持**: 最大支持100MB视频文件

### 🧠 AI情绪分析
- **人脸检测**: 自动检测视频中的人脸
- **情绪识别**: 识别8种基本情绪（中性、快乐、悲伤、惊讶、恐惧、厌恶、愤怒、蔑视）
- **置信度评估**: 提供每个情绪预测的置信度
- **统计分析**: 计算主要情绪、情绪分布和平均置信度

### 📊 实时结果展示
- **即时反馈**: 上传后立即开始分析并显示进度
- **详细报告**: 包含情绪分布、置信度、检测帧数等详细信息
- **可视化展示**: 使用表情符号和颜色编码增强用户体验
- **历史记录**: 侧边栏显示所有分析历史

## 技术架构

### 前端技术
- **HTML5 + JavaScript**: 现代Web界面
- **WebRTC**: 摄像头访问和视频录制
- **Socket.IO**: 实时通信
- **响应式设计**: 适配不同屏幕尺寸

### 后端技术
- **Flask**: Web框架
- **Flask-SocketIO**: WebSocket支持
- **PyTorch**: 深度学习框架
- **OpenCV**: 视频处理
- **EmoNet**: 情绪识别模型

## 安装与配置

### 1. 环境要求
```bash
# Python 3.7+
# CUDA (可选，用于GPU加速)
```

### 2. 安装依赖
```bash
# 基础依赖
pip install flask flask-socketio flask-cors
pip install torch torchvision torchaudio
pip install opencv-python numpy
pip install face-alignment

# 或使用requirements文件
pip install -r ../multimodal-emotion-recognition-system-master/requirements_camera.txt
```

### 3. 模型文件
确保EmoNet预训练模型文件位于正确路径：
```
../multimodal-emotion-recognition-system-master/emonet/pretrained/
├── emonet_8.pth
└── (其他模型文件)
```

## 使用方法

### 方法一：使用启动脚本（推荐）
```bash
cd VisionWeb
python start_emotion_server.py
```

### 方法二：直接启动
```bash
cd VisionWeb
python server.py
```

### 3. 访问系统
在浏览器中打开：`http://localhost:5000`

### 4. 使用流程
1. **录制视频**：点击"开始录制"按钮，对着摄像头录制
2. **上传视频**：或使用"手动上传"功能选择本地视频文件
3. **等待分析**：系统自动进行情绪分析
4. **查看结果**：在右侧面板查看详细的分析结果

## 分析结果说明

### 情绪类别
- 😐 **neutral**: 中性
- 😄 **happy**: 快乐
- 😢 **sad**: 悲伤
- 😮 **surprise**: 惊讶
- 😨 **fear**: 恐惧
- 🤢 **disgust**: 厌恶
- 😠 **anger**: 愤怒
- 😤 **contempt**: 蔑视

### 结果指标
- **主要情绪**: 视频中出现最频繁的情绪
- **平均置信度**: 所有预测的平均置信度（0-1）
- **检测帧数**: 成功检测到人脸的帧数
- **情绪分布**: 各种情绪的出现次数和百分比

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保有足够的内存和存储空间
   - 检查PyTorch安装是否正确

2. **摄像头无法访问**
   - 检查浏览器权限设置
   - 确保摄像头未被其他应用占用
   - 尝试使用HTTPS访问（某些浏览器要求）

3. **视频上传失败**
   - 检查文件大小（最大100MB）
   - 确保视频格式受支持
   - 检查网络连接

4. **情绪分析失败**
   - 确保视频中有清晰的人脸
   - 检查视频质量和光线条件
   - 查看服务器日志获取详细错误信息

### 性能优化

1. **GPU加速**
   ```python
   # 系统会自动检测并使用GPU（如果可用）
   # 确保安装了CUDA版本的PyTorch
   ```

2. **内存优化**
   - 对于长视频，系统只分析前几帧以节省内存
   - 可以调整批处理大小以适应硬件配置

## 开发说明

### 文件结构
```
VisionWeb/
├── server.py                          # 主服务器文件（已集成情绪识别）
├── start_emotion_server.py             # 启动脚本
├── templates/
│   └── index.html                      # 前端界面（已更新）
├── static/
│   ├── BackGround.jpg                  # 背景图片
│   └── logo.png                        # Logo
├── uploads/                            # 上传文件存储目录
└── README_EMOTION_INTEGRATION.md       # 本文档
```

### 自定义配置

可以在`server.py`中修改以下配置：

```python
# 情绪识别配置
EMOTION_CLASSES = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 256

# 服务器配置
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
```

## 版本历史

- **V2.0**: 集成AI情绪分析功能
- **V1.21**: 原始VisionWeb视频录制系统

## 开发团队

- 王肇麒
- 石垒
- 伍勤辉
- 王宇杰

## 许可证

本项目遵循相关开源许可证。请查看各组件的具体许可证要求。

---

🎉 **现在您可以享受完整的视频情绪分析体验了！**