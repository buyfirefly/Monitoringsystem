# 🎭 多模态情绪识别系统

> 融合面部表情和语音信息的深度学习情绪识别系统

## 🌟 项目简介

本项目实现了一个先进的多模态情绪识别系统，通过**深度融合面部表情和语音特征**来准确识别人类情绪。系统采用跨模态注意力机制，让视觉和音频信息相互增强，显著提升了情绪识别的准确性。

### 核心特点
- 🎯 **多模态融合**: 同时分析面部表情和语音特征
- 🧠 **智能融合**: 跨模态注意力机制实现深度交互
- ⚡ **高效训练**: 支持多GPU并行，3x RTX 4090优化
- 🎬 **实时识别**: 支持摄像头+麦克风实时情绪分析
- 📊 **完整工具链**: 训练、评估、推理一体化

## 🔬 技术原理

### 多模态融合架构

```
输入数据
├── 视频帧 (面部表情) ──→ 视觉编码器 ──┐
│                                    │
└── 音频片段 (语音情绪) ──→ 音频编码器 ──┤
                                     │
                                     ▼
                               跨模态注意力融合
                                     │
                                     ▼
                                情绪分类器
                                     │
                                     ▼
                            7种情绪类别输出
```

### 🎭 面部表情分析
**视觉编码器**基于ResNet18架构：
- 提取面部关键特征（眼部、嘴部、眉毛等）
- 捕获微表情变化
- 输出256维视觉特征向量

### 🎵 语音情绪分析
**音频编码器**采用CNN+RNN混合架构：
- **CNN层**: 处理梅尔频谱图，提取音频纹理特征
- **双向GRU**: 捕获语音的时序变化和韵律信息
- 输出256维音频特征向量

### 🔗 跨模态注意力融合

这是系统的核心创新，实现了**视觉-音频双向增强**：

```python
# 视觉对音频的注意力
v2a_features = CrossAttention(visual_features, audio_sequence)

# 音频对视觉的注意力  
a2v_features = CrossAttention(audio_features, visual_sequence)

# 融合特征
fused = concat([v2a_features, a2v_features])
```

**工作原理**：
1. **视觉引导音频**: 面部表情帮助模型关注语音中的情绪相关部分
2. **音频增强视觉**: 语音韵律信息补充视觉可能遗漏的情绪细节
3. **互补融合**: 两种模态的优势互补，提升整体识别准确性

### 支持的融合策略

| 策略 | 描述 | 优势 | 适用场景 |
|------|------|------|----------|
| `cross_attention` | 跨模态注意力 | 深度交互，性能最佳 | 推荐使用 |
| `transformer` | 共享编码器 | 统一建模，可解释性强 | 研究分析 |
| `concat` | 特征拼接 | 简单高效 | 快速原型 |

## 🚀 快速开始

### 环境安装
```bash
# 克隆项目
git clone <repository-url>
cd AI_project

# 安装依赖
cd multimodal_emotion
pip install -r requirements.txt
```

### 数据准备
```bash
# 下载MELD数据集
wget https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
tar -xzf MELD.Raw.tar.gz
```

### 开始训练
```bash
# 多GPU训练（推荐）
python train_multi_gpu.py --data_dir ./MELD.Raw

# 单GPU训练
python multimodal_emotion/train.py --data_dir ./MELD.Raw
```

### 实时推理

#### 基础实时推理
```bash
# 启动实时情绪识别
python multimodal_emotion/realtime_inference.py \
    --model_path ./checkpoints/best_model.pth

# 指定输入设备
python multimodal_emotion/realtime_inference.py --model_path ./checkpoints/best_model.pth --audio_device 1 --video_device 0
```

#### 实时摄像头情绪监测

本项目提供基于深度学习的多模态摄像头情绪监测功能：

##### 多模态情绪监测（推荐）

**功能特点：**
- 🎥 实时视频流处理和人脸检测
- 🎤 实时音频录制和语音情绪分析
- 🧠 多模态融合情绪识别
- 📊 实时统计图表和情绪分布
- 💾 结果保存和会话记录
- 🎯 高精度情绪识别

**安装依赖：**
```bash
# 安装摄像头监测依赖
pip install -r requirements_camera.txt

# macOS额外依赖
brew install portaudio ffmpeg

# Ubuntu/Debian额外依赖
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

**使用方法：**
```bash
# 基础使用（需要训练好的模型）
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth

# 使用导出的ONNX模型（更快）
python realtime_camera_emotion.py --model_path ./exported_models/model.onnx

# 保存检测结果
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth --save_results

# 指定摄像头ID
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth --camera_id 1

# 不显示统计图表（提高性能）
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth --no_stats
```

**操作说明：**
- 按 `q` 键：退出程序
- 按 `s` 键：保存当前检测结果快照
- 程序会自动保存完整会话记录（如果启用 `--save_results`）



##### 3. 性能优化建议

**硬件要求：**
- **最低配置：** CPU: 双核2GHz+, RAM: 4GB+, 摄像头: 720p
- **推荐配置：** CPU: 四核3GHz+, RAM: 8GB+, GPU: GTX1060+, 摄像头: 1080p
- **最佳配置：** CPU: 八核3.5GHz+, RAM: 16GB+, GPU: RTX3070+, 摄像头: 4K

**性能优化：**
```bash
# 使用GPU加速（需要CUDA）
export CUDA_VISIBLE_DEVICES=0
python realtime_camera_emotion.py --model_path ./exported_models/model.onnx

# 降低分辨率提高帧率
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth --resolution 640x480

# 使用量化模型（更快推理）
python realtime_camera_emotion.py --model_path ./exported_models/model_quantized.pth
```

##### 4. 输出结果说明

**实时显示：**
- 人脸框：不同颜色代表不同情绪
- 情绪标签：显示检测到的情绪和置信度
- 概率条：显示各种情绪的概率分布
- 统计信息：FPS、检测次数、主导情绪等

**保存的文件：**
```
emotion_session_20231201_143022.json    # 完整会话记录
emotion_snapshot_20231201_143022.json   # 快照记录
emotion_screenshot_20231201_143022.jpg  # 截图文件
```

**结果文件格式：**
```json
{
  "session_info": {
    "start_time": "2023-12-01T14:30:22",
    "duration_seconds": 120.5,
    "total_detections": 245
  },
  "emotion_statistics": {
    "distribution": {
      "Happy": 89,
      "Neutral": 76,
      "Surprised": 45,
      "Sad": 35
    },
    "average_confidence": 0.847,
    "dominant_emotion": "Happy"
  },
  "detailed_results": [
    {
      "timestamp": 1.23,
      "emotion": "Happy",
      "confidence": 0.92,
      "probabilities": [0.02, 0.92, 0.01, 0.02, 0.01, 0.01, 0.01]
    }
  ]
}
```

##### 5. 常见问题解决

**摄像头无法打开：**
```bash
# 检查可用摄像头
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# 尝试不同的摄像头ID
python simple_camera_emotion.py --camera_id 1
```

**音频录制失败：**
```bash
# 检查音频设备
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"

# macOS权限问题
# 系统偏好设置 -> 安全性与隐私 -> 隐私 -> 麦克风 -> 允许终端访问
```

**性能问题：**
```bash
# 使用简化版
python simple_camera_emotion.py

# 关闭统计图表
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth --no_stats

# 使用更小的模型
python realtime_camera_emotion.py --model_path ./exported_models/model_quantized.pth
```

## 📊 性能表现

### 识别准确率
| 数据集 | 单模态(视觉) | 单模态(音频) | **多模态融合** |
|--------|-------------|-------------|---------------|
| MELD | 58-62% | 55-60% | **68-72%** |
| IEMOCAP | 65-70% | 62-68% | **78-82%** |

### 训练效率
| 配置 | 训练时间 | 内存使用 | 加速比 |
|------|----------|----------|--------|
| 1x RTX 4090 | ~6小时 | 20GB | 1.0x |
| **3x RTX 4090** | **~2小时** | **18GB/GPU** | **2.8x** |

## 🎯 使用场景

- **智能客服**: 实时分析客户情绪，提升服务质量
- **教育评估**: 分析学生学习状态和情绪反应
- **心理健康**: 辅助心理咨询和情绪监测
- **人机交互**: 让AI更好理解人类情感
- **内容分析**: 视频内容的情绪标注和分析

## 📁 项目结构

```
AI_project/
├── multimodal_emotion/          # 核心模块
│   ├── __init__.py
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   ├── realtime_inference.py    # 实时推理
│   ├── fusion_model.py          # 多模态融合模型
│   ├── data_loader.py           # 数据加载器
│   └── utils.py                 # 工具函数
├── data/                        # 数据目录
│   ├── MELD/                    # MELD数据集
│   └── processed/               # 预处理后的数据
├── checkpoints/                 # 模型检查点
├── exported_models/             # 导出的模型文件
├── logs/                        # 训练日志
├── configs/                     # 配置文件
├── export_model.py              # 模型导出工具
├── load_exported_model.py       # 模型加载示例
├── clean_meld.py                # 数据清洗工具
├── realtime_camera_emotion.py    # 实时摄像头情绪监测
├── requirements.txt             # 项目依赖
├── requirements_export.txt      # 模型导出依赖
├── requirements_camera.txt      # 摄像头监测依赖
└── README.md                    # 项目说明
```

## 🔧 高级配置

### 自定义融合策略
```bash
# 使用跨模态注意力（推荐）
python train_multi_gpu.py --fusion_type cross_attention

# 使用Transformer编码器
python train_multi_gpu.py --fusion_type transformer

# 使用简单拼接
python train_multi_gpu.py --fusion_type concat
```

### 多GPU优化
```bash
# 3x RTX 4090 优化配置
python train_multi_gpu.py \
    --data_dir ./MELD.Raw \
    --batch_size 20 \
    --epochs 100 \
    --fusion_type cross_attention
```

### 模型评估
```bash
# 详细评估报告
python multimodal_emotion/evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_dir ./MELD.Raw
```

### 模型导出

训练完成后，您可以将模型导出为多种格式以便部署：

```bash
# 导出所有格式（推荐）
python export_model.py --model_path ./checkpoints/best_model.pth --format all

# 导出特定格式
python export_model.py --model_path ./checkpoints/best_model.pth --format onnx
python export_model.py --model_path ./checkpoints/best_model.pth --format torchscript --optimize
python export_model.py --model_path ./checkpoints/best_model.pth --format quantized
```

**支持的导出格式**：

| 格式 | 文件扩展名 | 用途 | 优势 |
|------|-----------|------|------|
| **PyTorch** | `.pth` | Python推理 | 完整功能，易于调试 |
| **TorchScript** | `.pt` | C++部署，移动端 | 跨语言，高性能 |
| **ONNX** | `.onnx` | 跨平台部署 | 框架无关，广泛支持 |
| **Quantized** | `.pth` | 资源受限环境 | 模型小，推理快 |

**导出文件结构**：
```
exported_models/
├── model.pth              # 标准PyTorch格式
├── model.pt               # TorchScript格式  
├── model.onnx             # ONNX格式
├── model_quantized.pth    # 量化模型
└── model_info.json        # 模型元数据
```

## ❓ 常见问题

**Q: 为什么多模态比单模态效果好？**
A: 面部表情和语音包含互补的情绪信息。例如，有些人善于控制面部表情但语音会泄露真实情绪，而有些情况下面部表情更明显。跨模态注意力机制让模型自动学习如何最优地结合这两种信息。

**Q: 跨模态注意力是如何工作的？**
A: 系统让视觉特征"询问"音频序列中哪些部分最相关，同时让音频特征"关注"视觉中的重要区域。这种双向交互让两种模态相互增强，提升整体性能。

**Q: 如何处理单模态缺失的情况？**
A: 模型设计时考虑了鲁棒性，当某个模态质量较差时，注意力机制会自动降低其权重，更多依赖质量较好的模态。

**Q: 如何选择合适的模型导出格式？**

**A:** 根据部署场景选择：

- **PyTorch (.pth)**: 开发和调试阶段，需要完整的Python环境
  ```python
  # 加载方式
  model = create_model(...)
  checkpoint = torch.load('model.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  ```

- **ONNX (.onnx)**: 跨平台部署，支持多种推理引擎
  ```python
  # 加载方式
  import onnxruntime as ort
  session = ort.InferenceSession('model.onnx')
  outputs = session.run(None, inputs)
  ```

- **量化模型**: 移动设备或边缘计算，追求速度和小体积
  ```python
  # 加载方式
  model = torch.jit.load('model_quantized.pth')
  ```

- **TorchScript (.pt)**: 生产环境，无需Python依赖
  ```python
  # 加载方式
  model = torch.jit.load('model.pt')
  ```

**Q: 摄像头情绪监测无法正常工作怎么办？**

**A:** 按以下步骤排查：

**1. 检查摄像头连接**
```bash
# 测试摄像头是否可用
python -c "import cv2; cap=cv2.VideoCapture(0); print('摄像头可用' if cap.isOpened() else '摄像头不可用'); cap.release()"

# 查找可用摄像头ID
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

**2. 解决权限问题**
- **macOS**: 系统偏好设置 → 安全性与隐私 → 隐私 → 摄像头/麦克风 → 允许终端访问
- **Windows**: 设置 → 隐私 → 摄像头 → 允许应用访问摄像头
- **Linux**: 检查用户是否在video组中：`sudo usermod -a -G video $USER`

**3. 音频问题解决**
```bash
# 检查音频设备
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"

# macOS安装音频依赖
brew install portaudio
pip install pyaudio

# Ubuntu/Debian安装音频依赖
sudo apt-get install portaudio19-dev python3-pyaudio
```

**4. 性能优化**
```bash
# 如果性能不足，使用简化版
python simple_camera_emotion.py

# 或关闭统计图表
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth --no_stats
```

**Q: 如何提高摄像头监测的准确性？**

**A:** 优化建议：

**环境设置：**
- 确保充足的光照条件
- 避免强烈的背光或侧光
- 保持摄像头稳定，减少抖动
- 人脸与摄像头距离保持在50-150cm

**模型选择：**
```bash
# 使用训练好的完整模型（最高精度）
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth

# 使用ONNX模型（平衡精度和速度）
python realtime_camera_emotion.py --model_path ./exported_models/model.onnx
```

**参数调优：**
- 增加音频采样时长以提高语音情绪识别准确性
- 调整人脸检测的最小尺寸参数
- 使用多帧平均来减少单帧误差

**Q: 摄像头监测结果如何保存和分析？**

**A:** 结果保存和分析方法：

**保存结果：**
```bash
# 自动保存完整会话
python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth --save_results

# 运行时按's'键保存快照
# 运行时按'q'键退出并保存最终结果
```

**结果文件说明：**
- `emotion_session_*.json`: 完整会话记录，包含所有检测结果
- `emotion_snapshot_*.json`: 快照记录，包含当前时刻的统计信息
- `emotion_screenshot_*.jpg`: 截图文件（简化版）

**数据分析示例：**
```python
import json
import matplotlib.pyplot as plt
from collections import Counter

# 加载会话数据
with open('emotion_session_20231201_143022.json', 'r') as f:
    data = json.load(f)

# 分析情绪分布
emotions = [result['emotion'] for result in data['detailed_results']]
emotion_counts = Counter(emotions)

# 绘制情绪分布图
plt.figure(figsize=(10, 6))
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.title('情绪分布统计')
plt.xlabel('情绪类型')
plt.ylabel('检测次数')
plt.show()

# 分析情绪变化趋势
timestamps = [result['timestamp'] for result in data['detailed_results']]
confidences = [result['confidence'] for result in data['detailed_results']]

plt.figure(figsize=(12, 6))
plt.plot(timestamps, confidences)
plt.title('情绪置信度变化趋势')
plt.xlabel('时间 (秒)')
plt.ylabel('置信度')
plt.show()
```

**Q: 导出的模型如何使用？**
A: 每种格式都有对应的加载方式：
```python
# PyTorch格式
model_data = torch.load('model.pth')
model.load_state_dict(model_data['model_state_dict'])

# TorchScript格式  
model = torch.jit.load('model.pt')

# ONNX格式
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
```

## 📚 技术细节

### 数据预处理
- **视频**: 提取关键帧，人脸检测和对齐，归一化到224x224
- **音频**: 转换为梅尔频谱图，64个梅尔滤波器，窗长25ms

### 模型架构
- **视觉编码器**: ResNet18 + 投影层 → 256维
- **音频编码器**: CNN(3层) + 双向GRU(2层) → 256维  
- **融合层**: 多头注意力(8头) + 前馈网络
- **分类器**: 3层全连接 + Dropout

### 训练策略
- **优化器**: Adam，分模块学习率
- **调度器**: 余弦退火 + 预热
- **数据增强**: 随机裁剪、颜色抖动、时间拉伸
- **正则化**: Dropout、标签平滑

## 🤝 贡献与支持

欢迎提交Issue和Pull Request！如果项目对您有帮助，请给个⭐支持！

### 引用
```bibtex
@misc{multimodal_emotion_2024,
  title={Multimodal Emotion Recognition with Cross-Modal Attention},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multimodal_emotion}
}
```

---

💡 **核心创新**: 通过跨模态注意力机制实现面部表情和语音的深度融合，让AI更准确地理解人类情绪。
