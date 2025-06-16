#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出模型使用示例
-------------------------------------------------
演示如何加载和使用不同格式的导出模型进行推理

使用示例:
  python examples/load_exported_model.py --model_path ./exported_models/model.pth --format pytorch
  python examples/load_exported_model.py --model_path ./exported_models/model.onnx --format onnx
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Tuple

try:
    import onnxruntime as ort
except ImportError:
    print("Warning: onnxruntime not installed. ONNX models cannot be loaded.")
    ort = None


class ModelLoader:
    """模型加载器 - 支持多种格式"""
    
    def __init__(self, model_path: str, format_type: str = 'auto'):
        self.model_path = Path(model_path)
        self.format_type = format_type if format_type != 'auto' else self._detect_format()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 情绪标签
        self.emotion_labels = [
            'neutral', 'happiness', 'sadness', 'anger', 
            'fear', 'disgust', 'surprise'
        ]
        
        print(f"🔍 检测到模型格式: {self.format_type}")
        print(f"🖥️  使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        
    def _detect_format(self) -> str:
        """自动检测模型格式"""
        suffix = self.model_path.suffix.lower()
        if suffix == '.onnx':
            return 'onnx'
        elif suffix == '.pt':
            return 'torchscript'
        elif suffix == '.pth':
            return 'pytorch'
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _load_model(self):
        """根据格式加载模型"""
        if self.format_type == 'pytorch':
            return self._load_pytorch_model()
        elif self.format_type == 'torchscript':
            return self._load_torchscript_model()
        elif self.format_type == 'onnx':
            return self._load_onnx_model()
        else:
            raise ValueError(f"不支持的模型格式: {self.format_type}")
    
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        print("📦 加载PyTorch模型...")
        
        # 加载模型数据
        model_data = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 检查是否是导出的格式
        if 'model_state_dict' in model_data:
            # 这里需要重新创建模型架构
            # 在实际使用中，您需要导入并创建相应的模型
            print("⚠️  注意: 需要模型架构代码来完全加载PyTorch模型")
            print(f"📋 模型配置: {model_data.get('config', {})}")
            return model_data
        else:
            # 直接加载的模型
            return model_data
    
    def _load_torchscript_model(self):
        """加载TorchScript模型"""
        print("📦 加载TorchScript模型...")
        model = torch.jit.load(self.model_path, map_location=self.device)
        model.eval()
        return model
    
    def _load_onnx_model(self):
        """加载ONNX模型"""
        if ort is None:
            raise ImportError("请安装onnxruntime: pip install onnxruntime")
        
        print("📦 加载ONNX模型...")
        
        # 设置执行提供者
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        # 打印模型信息
        print(f"📊 输入节点: {[inp.name for inp in session.get_inputs()]}")
        print(f"📊 输出节点: {[out.name for out in session.get_outputs()]}")
        
        return session
    
    def predict(self, visual_input: np.ndarray, audio_input: np.ndarray) -> Tuple[np.ndarray, str]:
        """进行预测"""
        start_time = time.time()
        
        if self.format_type == 'pytorch':
            # PyTorch格式需要模型架构，这里只是示例
            print("⚠️  PyTorch模型需要完整的模型架构代码")
            # 返回随机结果作为示例
            logits = np.random.randn(7)
        
        elif self.format_type == 'torchscript':
            # TorchScript推理
            visual_tensor = torch.from_numpy(visual_input).float().to(self.device)
            audio_tensor = torch.from_numpy(audio_input).float().to(self.device)
            
            with torch.no_grad():
                logits = self.model(visual_tensor, audio_tensor)
                logits = logits.cpu().numpy().squeeze()
        
        elif self.format_type == 'onnx':
            # ONNX推理
            inputs = {
                'visual_input': visual_input.astype(np.float32),
                'audio_input': audio_input.astype(np.float32)
            }
            outputs = self.model.run(None, inputs)
            logits = outputs[0].squeeze()
        
        # 计算概率和预测类别
        probabilities = self._softmax(logits)
        predicted_class = np.argmax(probabilities)
        predicted_emotion = self.emotion_labels[predicted_class]
        confidence = probabilities[predicted_class]
        
        inference_time = time.time() - start_time
        
        print(f"⚡ 推理时间: {inference_time*1000:.2f}ms")
        print(f"🎯 预测情绪: {predicted_emotion} (置信度: {confidence:.3f})")
        
        return probabilities, predicted_emotion
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算softmax概率"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def benchmark(self, num_runs: int = 100):
        """性能基准测试"""
        print(f"\n🏃 开始性能测试 ({num_runs} 次推理)...")
        
        # 创建随机输入
        visual_input = np.random.randn(1, 3, 224, 224)
        audio_input = np.random.randn(1, 64, 500)
        
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            self.predict(visual_input, audio_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 统计结果
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n📊 性能统计:")
        print(f"   平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   最快推理时间: {min_time:.2f} ms")
        print(f"   最慢推理时间: {max_time:.2f} ms")
        print(f"   吞吐量: {1000/avg_time:.1f} FPS")


def create_dummy_inputs() -> Tuple[np.ndarray, np.ndarray]:
    """创建示例输入数据"""
    # 模拟面部图像 (batch_size=1, channels=3, height=224, width=224)
    visual_input = np.random.randn(1, 3, 224, 224)
    
    # 模拟音频梅尔频谱图 (batch_size=1, n_mels=64, time_steps=500)
    audio_input = np.random.randn(1, 64, 500)
    
    return visual_input, audio_input


def main():
    parser = argparse.ArgumentParser(description='Load and test exported models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to exported model')
    parser.add_argument('--format', type=str, choices=['pytorch', 'torchscript', 'onnx', 'auto'],
                       default='auto', help='Model format')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model_path).exists():
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    try:
        # 加载模型
        loader = ModelLoader(args.model_path, args.format)
        
        # 创建示例输入
        visual_input, audio_input = create_dummy_inputs()
        
        print(f"\n🎬 开始推理测试...")
        print(f"📊 输入形状: 视觉={visual_input.shape}, 音频={audio_input.shape}")
        
        # 进行预测
        probabilities, predicted_emotion = loader.predict(visual_input, audio_input)
        
        # 显示详细结果
        print(f"\n📈 详细预测结果:")
        for i, (emotion, prob) in enumerate(zip(loader.emotion_labels, probabilities)):
            print(f"   {emotion:>10}: {prob:.3f} {'🎯' if i == np.argmax(probabilities) else ''}")
        
        # 性能测试
        if args.benchmark:
            loader.benchmark(args.num_runs)
        
        print(f"\n✅ 测试完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()