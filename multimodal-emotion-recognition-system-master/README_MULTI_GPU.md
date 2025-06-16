# 多GPU训练指南 - 3x RTX 4090优化

本指南介绍如何使用3个RTX 4090 GPU进行多模态情感识别模型的训练。

## 🚀 快速开始

### 1. 使用优化的训练脚本

```bash
# 使用默认优化参数
python train_multi_gpu.py --data_dir /path/to/your/dataset

# 自定义参数
python train_multi_gpu.py \
    --data_dir /path/to/your/dataset \
    --dataset MELD \
    --batch_size 20 \
    --epochs 150
```

### 2. 使用原始训练脚本

```bash
python multimodal_emotion/train.py \
    --data_dir /path/to/your/dataset \
    --use_multi_gpu \
    --gpu_ids 0 1 2 \
    --batch_size 16 \
    --num_workers 12 \
    --auto_scale_lr
```

## ⚙️ 优化配置

### GPU配置
- **GPU数量**: 3个 RTX 4090 (24GB each)
- **总显存**: 72GB
- **并行策略**: DataParallel

### 批次大小优化
- **推荐配置**: 每GPU 20-24个样本
- **总批次大小**: 60-72个样本
- **内存使用**: 约18-20GB per GPU

### 学习率调整
- **自动缩放**: 学习率 × GPU数量
- **Visual Encoder**: 3e-4
- **Audio Encoder**: 3e-4  
- **Fusion Layer**: 3e-3

### 数据加载优化
- **Workers**: 12个 (4 per GPU)
- **Pin Memory**: 启用
- **Non-blocking Transfer**: 启用

## 📊 性能监控

### TensorBoard监控
```bash
tensorboard --logdir ./logs_multi_gpu
```

监控指标:
- 训练/验证损失和准确率
- 每个GPU的内存使用情况
- 学习率变化
- 梯度范数

### GPU内存监控
```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 或使用gpustat
pip install gpustat
gpustat -i 1
```

## 🔧 故障排除

### 常见问题

#### 1. 内存不足 (OOM)
```bash
# 减小批次大小
python train_multi_gpu.py --data_dir /path/to/dataset --batch_size 16

# 或减少workers
python train_multi_gpu.py --data_dir /path/to/dataset --num_workers 8
```

#### 2. GPU利用率低
- 检查数据加载是否成为瓶颈
- 增加`num_workers`
- 确保`pin_memory=True`

#### 3. 训练速度慢
```bash
# 启用cuDNN benchmark
export TORCH_CUDNN_V8_API_ENABLED=1

# 优化内存分配
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### 4. 数据加载错误
确保已应用错误处理修复:
- `data_utils.py`中的视频读取错误处理
- `fusion_model.py`中的ResNet18权重警告修复

## 📈 性能基准

### 预期性能 (3x RTX 4090)
- **训练速度**: ~2-3秒/epoch (MELD数据集)
- **内存使用**: 18-20GB per GPU
- **GPU利用率**: 85-95%
- **数据加载**: <5% 等待时间

### 与单GPU对比
- **速度提升**: 2.7-2.9x
- **内存效率**: 更好的批次大小
- **训练稳定性**: 更大的有效批次大小

## 🎯 最佳实践

### 1. 批次大小选择
```python
# 根据模型复杂度调整
# 简单模型: 24-28 per GPU
# 复杂模型: 16-20 per GPU
# 超大模型: 8-12 per GPU
```

### 2. 学习率调度
```python
# 推荐使用余弦退火
--scheduler cosine

# 或使用ReduceLROnPlateau
--scheduler plateau --patience 10 --factor 0.5
```

### 3. 梯度累积 (可选)
如果需要更大的有效批次大小:
```python
# 在train.py中添加梯度累积
accumulation_steps = 2  # 有效批次大小 = batch_size * num_gpus * accumulation_steps
```

### 4. 混合精度训练 (可选)
```python
# 使用AMP加速训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images, mel_specs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📝 配置文件示例

### config_3x4090.json
```json
{
    "use_multi_gpu": true,
    "gpu_ids": [0, 1, 2],
    "batch_size": 22,
    "num_workers": 12,
    "lr_visual": 3e-4,
    "lr_audio": 3e-4,
    "lr_fusion": 3e-3,
    "auto_scale_lr": false,
    "fusion_type": "attention",
    "scheduler": "cosine",
    "grad_clip": 1.0,
    "epochs": 100
}
```

## 🔍 调试技巧

### 1. 验证多GPU设置
```python
import torch
print(f"可用GPU: {torch.cuda.device_count()}")
print(f"当前GPU: {torch.cuda.current_device()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

### 2. 检查模型并行
```python
# 在训练脚本中添加
if hasattr(model, 'module'):
    print("模型已使用DataParallel包装")
else:
    print("模型未使用并行")
```

### 3. 监控数据流
```python
# 添加到训练循环
print(f"Batch shape: images={images.shape}, audio={mel_specs.shape}")
print(f"Device: {images.device}")
```

## 📞 支持

如果遇到问题:
1. 检查GPU驱动和CUDA版本兼容性
2. 确认PyTorch版本支持你的CUDA版本
3. 验证数据集路径和格式
4. 查看错误日志和GPU内存使用情况

---

**注意**: 本配置专为3x RTX 4090优化，其他GPU配置可能需要调整参数。