#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出脚本
-------------------------------------------------
功能:
  1. 导出训练好的多模态情绪识别模型
  2. 支持多种格式: ONNX, TorchScript, 标准PyTorch
  3. 模型优化和量化
  4. 导出配置和元数据

使用示例:
  python export_model.py --model_path ./checkpoints/best_model.pth --format onnx
  python export_model.py --model_path ./checkpoints/best_model.pth --format torchscript --optimize
"""

import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple
import warnings

# 导入模型
from multimodal_emotion.fusion_model import create_model


class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model_path: str, output_dir: str = "./exported_models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型和配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.config = self._load_model()
        
        print(f"模型加载成功: {model_path}")
        print(f"模型配置: {self.config}")
        
    def _load_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """加载训练好的模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 获取配置
        config = checkpoint.get('config', {
            'fusion_type': 'cross_attention',
            'visual_dim': 256,
            'audio_dim': 256,
            'hidden_dim': 512,
            'num_classes': 7,
            'dropout': 0.1
        })
        
        # 创建模型
        model = create_model(
            fusion_type=config.get('fusion_type', 'cross_attention'),
            visual_dim=config.get('visual_dim', 256),
            audio_dim=config.get('audio_dim', 256),
            hidden_dim=config.get('hidden_dim', 512),
            num_classes=config.get('num_classes', 7),
            dropout=config.get('dropout', 0.1)
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def export_pytorch(self, filename: str = "model.pth") -> str:
        """导出标准PyTorch格式"""
        output_path = self.output_dir / filename
        
        # 保存完整模型信息
        export_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_class': 'MultimodalEmotionModel',
            'input_shapes': {
                'visual': [3, 224, 224],  # RGB图像
                'audio': [64, 500]        # 梅尔频谱图
            },
            'output_shape': [self.config.get('num_classes', 7)],
            'emotion_labels': ['neutral', 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
        }
        
        torch.save(export_data, output_path)
        print(f"✅ PyTorch模型已导出: {output_path}")
        return str(output_path)
    
    def export_torchscript(self, filename: str = "model.pt", optimize: bool = False) -> str:
        """导出TorchScript格式"""
        output_path = self.output_dir / filename
        
        # 创建示例输入
        batch_size = 1
        visual_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        audio_input = torch.randn(batch_size, 64, 500).to(self.device)
        
        try:
            # 使用torch.jit.trace进行追踪
            with torch.no_grad():
                traced_model = torch.jit.trace(self.model, (visual_input, audio_input))
            
            # 可选优化
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)
                print("🚀 TorchScript模型已优化")
            
            # 保存模型
            traced_model.save(str(output_path))
            
            # 验证导出的模型
            loaded_model = torch.jit.load(str(output_path))
            with torch.no_grad():
                original_output = self.model(visual_input, audio_input)
                traced_output = loaded_model(visual_input, audio_input)
                diff = torch.abs(original_output - traced_output).max().item()
                print(f"📊 模型验证: 最大差异 = {diff:.6f}")
            
            print(f"✅ TorchScript模型已导出: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ TorchScript导出失败: {e}")
            return None
    
    def export_onnx(self, filename: str = "model.onnx", opset_version: int = 11) -> str:
        """导出ONNX格式"""
        output_path = self.output_dir / filename
        
        # 创建示例输入
        batch_size = 1
        visual_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        audio_input = torch.randn(batch_size, 64, 500).to(self.device)
        
        try:
            # 导出ONNX
            torch.onnx.export(
                self.model,
                (visual_input, audio_input),
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['visual_input', 'audio_input'],
                output_names=['emotion_logits'],
                dynamic_axes={
                    'visual_input': {0: 'batch_size'},
                    'audio_input': {0: 'batch_size'},
                    'emotion_logits': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # 测试推理
            ort_session = ort.InferenceSession(str(output_path))
            ort_inputs = {
                'visual_input': visual_input.cpu().numpy(),
                'audio_input': audio_input.cpu().numpy()
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # 比较输出
            with torch.no_grad():
                torch_output = self.model(visual_input, audio_input).cpu().numpy()
            diff = np.abs(torch_output - ort_outputs[0]).max()
            print(f"📊 ONNX验证: 最大差异 = {diff:.6f}")
            
            print(f"✅ ONNX模型已导出: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ ONNX导出失败: {e}")
            return None
    
    def export_quantized(self, filename: str = "model_quantized.pth") -> str:
        """导出量化模型（减小模型大小）"""
        output_path = self.output_dir / filename
        
        try:
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                self.model.cpu(),
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            # 保存量化模型
            export_data = {
                'model': quantized_model,
                'config': self.config,
                'quantized': True,
                'emotion_labels': ['neutral', 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
            }
            
            torch.save(export_data, output_path)
            
            # 计算模型大小
            original_size = self.model_path.stat().st_size / (1024 * 1024)  # MB
            quantized_size = output_path.stat().st_size / (1024 * 1024)  # MB
            compression_ratio = original_size / quantized_size
            
            print(f"📦 模型量化完成:")
            print(f"   原始大小: {original_size:.2f} MB")
            print(f"   量化大小: {quantized_size:.2f} MB")
            print(f"   压缩比: {compression_ratio:.2f}x")
            print(f"✅ 量化模型已导出: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 模型量化失败: {e}")
            return None
    
    def export_metadata(self, filename: str = "model_info.json") -> str:
        """导出模型元数据"""
        output_path = self.output_dir / filename
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        metadata = {
            'model_info': {
                'name': 'Multimodal Emotion Recognition Model',
                'version': '1.0.0',
                'framework': 'PyTorch',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024)
            },
            'config': self.config,
            'input_specs': {
                'visual': {
                    'shape': [3, 224, 224],
                    'dtype': 'float32',
                    'description': 'RGB face image, normalized to [0,1]'
                },
                'audio': {
                    'shape': [64, 500],
                    'dtype': 'float32', 
                    'description': 'Mel-spectrogram features'
                }
            },
            'output_specs': {
                'emotion_logits': {
                    'shape': [7],
                    'dtype': 'float32',
                    'description': 'Emotion classification logits'
                }
            },
            'emotion_labels': {
                0: 'neutral',
                1: 'happiness', 
                2: 'sadness',
                3: 'anger',
                4: 'fear',
                5: 'disgust',
                6: 'surprise'
            },
            'preprocessing': {
                'visual': {
                    'resize': [224, 224],
                    'normalize': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    }
                },
                'audio': {
                    'sample_rate': 16000,
                    'n_mels': 64,
                    'hop_length': 512,
                    'win_length': 1024
                }
            },
            'performance': {
                'dataset': 'MELD',
                'accuracy': 'See training logs',
                'inference_time': 'Depends on hardware'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 模型元数据已导出: {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Export Multimodal Emotion Recognition Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./exported_models',
                       help='Output directory for exported models')
    parser.add_argument('--format', type=str, choices=['pytorch', 'torchscript', 'onnx', 'quantized', 'all'],
                       default='all', help='Export format')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize exported model (for TorchScript)')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not Path(args.model_path).exists():
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    # 创建导出器
    exporter = ModelExporter(args.model_path, args.output_dir)
    
    print(f"\n🚀 开始导出模型...")
    print(f"📁 输出目录: {args.output_dir}")
    
    exported_files = []
    
    # 根据格式导出
    if args.format in ['pytorch', 'all']:
        file_path = exporter.export_pytorch()
        if file_path:
            exported_files.append(file_path)
    
    if args.format in ['torchscript', 'all']:
        file_path = exporter.export_torchscript(optimize=args.optimize)
        if file_path:
            exported_files.append(file_path)
    
    if args.format in ['onnx', 'all']:
        file_path = exporter.export_onnx(opset_version=args.opset_version)
        if file_path:
            exported_files.append(file_path)
    
    if args.format in ['quantized', 'all']:
        file_path = exporter.export_quantized()
        if file_path:
            exported_files.append(file_path)
    
    # 总是导出元数据
    metadata_path = exporter.export_metadata()
    if metadata_path:
        exported_files.append(metadata_path)
    
    print(f"\n✅ 导出完成! 共导出 {len(exported_files)} 个文件:")
    for file_path in exported_files:
        print(f"   📄 {file_path}")
    
    print(f"\n💡 使用提示:")
    print(f"   - PyTorch: 用于Python推理")
    print(f"   - TorchScript: 用于C++部署或移动端")
    print(f"   - ONNX: 用于跨平台部署")
    print(f"   - Quantized: 用于资源受限环境")


if __name__ == '__main__':
    main()