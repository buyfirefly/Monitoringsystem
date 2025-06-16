#!/usr/bin/env python3
"""
多GPU训练脚本 - 专为3x RTX 4090优化

使用方法:
python train_multi_gpu.py --data_dir /path/to/dataset

特性:
- 自动检测和使用3个GPU
- 优化的batch size和学习率
- 混合精度训练
- 梯度累积
- 内存优化
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent / 'multimodal_emotion'))

from multimodal_emotion.train import main as train_main

def setup_optimal_args():
    """为3x RTX 4090设置最优参数"""
    parser = argparse.ArgumentParser(description='Multi-GPU Training for 3x RTX 4090')
    
    # ==================== 必需参数 ====================
    # 数据集根目录，必须由用户指定
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='数据集根目录路径，包含训练、验证和测试数据')
    
    # ==================== 数据集参数 ====================
    # 选择要使用的数据集类型
    parser.add_argument('--dataset', type=str, default='MELD', 
                       choices=['MELD', 'IEMOCAP'],
                       help='数据集类型：MELD(多模态情绪对话数据集)或IEMOCAP(交互式情绪动作捕捉数据集)')
    
    # 优化的训练参数
    parser.add_argument('--batch_size', type=int, default=24, 
                       help='每个GPU的批次大小(24 * 3 GPU = 72总批次大小)，针对RTX 4090的24GB显存优化')
    
    # 训练轮数，100轮通常足够模型收敛
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练总轮数，通常100轮足够模型收敛')
    
    # 数据加载工作进程数，每个GPU分配4个worker
    parser.add_argument('--num_workers', type=int, default=12, 
                       help='数据加载工作进程数(每GPU 4个worker)，平衡CPU和GPU利用率')
    
    # ==================== 学习率设置 ====================
    # 视觉编码器学习率，较小以保持预训练特征
    parser.add_argument('--lr_visual', type=float, default=1e-4,
                       help='视觉编码器学习率，较小值保持预训练特征稳定性')
    
    # 音频编码器学习率，与视觉编码器相同
    parser.add_argument('--lr_audio', type=float, default=1e-4,
                       help='音频编码器学习率，与视觉编码器保持一致') 
    
    # 融合层学习率，较大以快速学习跨模态特征
    parser.add_argument('--lr_fusion', type=float, default=1e-3,
                       help='融合层学习率，较大值以快速学习跨模态特征融合')
    
    # ==================== GPU配置 ====================
    # 指定使用的GPU设备ID
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2],
                       help='使用的GPU设备ID列表，默认使用前3块GPU')
    
    # 启用多GPU训练模式
    parser.add_argument('--use_multi_gpu', action='store_true', default=True,
                       help='启用多GPU并行训练，提高训练效率')
    
    # 禁用自动学习率缩放，因为已手动设置合适的学习率
    parser.add_argument('--auto_scale_lr', action='store_true', default=False,
                       help='自动根据GPU数量缩放学习率，已手动优化故禁用')
    
    # ==================== 模型架构参数 ====================
    # 多模态融合策略选择
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                       choices=['cross_attention', 'concat', 'transformer'],
                       help='多模态融合策略：cross_attention(跨模态注意力)、concat(简单拼接)、transformer(Transformer融合)')
    
    # 视觉特征维度，通常为预训练模型输出维度
    parser.add_argument('--visual_dim', type=int, default=512,
                       help='视觉特征维度，对应视觉编码器输出特征的维度')
    
    # 音频特征维度，根据音频编码器确定
    parser.add_argument('--audio_dim', type=int, default=256,
                       help='音频特征维度，对应音频编码器输出特征的维度')
    
    # 融合后特征维度，影响最终分类器输入
    parser.add_argument('--fusion_dim', type=int, default=256,
                       help='融合后特征维度，决定融合层输出和分类器输入的维度')
    
    # 情绪分类类别数，根据数据集确定
    parser.add_argument('--num_classes', type=int, default=7,
                       help='情绪分类类别数，MELD数据集有7种情绪类别')
    
    # Dropout比率，防止过拟合
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比率，防止模型过拟合，0.3是经验最优值')
    
    # ==================== 优化器参数 ====================
    # L2正则化权重衰减系数
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减系数(L2正则化)，防止模型过拟合')
    
    # 梯度裁剪阈值，防止梯度爆炸
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪阈值，防止训练过程中梯度爆炸')
    
    # ==================== 学习率调度器参数 ====================
    # 学习率调度策略选择
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['plateau', 'cosine', 'step'],
                       help='学习率调度策略：plateau(性能平台期)、cosine(余弦退火)、step(阶梯衰减)')
    
    # ReduceLROnPlateau调度器的耐心值
    parser.add_argument('--patience', type=int, default=10,
                       help='学习率调度器耐心值，连续多少轮无改善后调整学习率')
    
    # 学习率衰减因子
    parser.add_argument('--factor', type=float, default=0.5,
                       help='学习率衰减因子，每次调整时学习率乘以该值')
    
    # ==================== 模型保存和日志 ====================
    # 模型检查点保存目录
    parser.add_argument('--save_dir', type=str, default='./checkpoints_multi_gpu',
                       help='模型检查点保存目录，存储训练过程中的模型权重')
    
    # 训练日志保存目录
    parser.add_argument('--log_dir', type=str, default='./logs_multi_gpu',
                       help='训练日志保存目录，存储损失值、准确率等训练指标')
    
    # 模型保存频率
    parser.add_argument('--save_freq', type=int, default=5,
                       help='模型保存频率，每隔多少个epoch保存一次检查点')
    
    # ==================== 训练恢复 ====================
    # 从检查点恢复训练
    parser.add_argument('--resume', type=str, default=None, 
                       help='从指定检查点文件恢复训练，用于中断后继续训练')
    
    # ==================== 训练控制参数 ====================
    # 只保存最佳模型
    parser.add_argument('--save_best_only', action='store_true', default=True,
                       help='只保存验证集上性能最佳的模型，节省存储空间')
    
    # 启用早停机制
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='启用早停机制，防止过拟合并节省训练时间')
    
    # 早停耐心值
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='早停耐心值，连续多少轮验证性能无改善后停止训练')
    
    # 梯度裁剪阈值(重复参数，与grad_clip相同)
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='梯度裁剪阈值，与grad_clip功能相同')
    
    # 启用自动混合精度训练
    parser.add_argument('--use_amp', action='store_true', default=True, 
                       help='启用自动混合精度(AMP)训练，提高训练速度并节省显存')
    
    # 日志打印间隔
    parser.add_argument('--log_interval', type=int, default=10,
                       help='训练日志打印间隔，每隔多少个batch打印一次训练信息')
    
    # 验证间隔
    parser.add_argument('--val_interval', type=int, default=1,
                       help='验证间隔，每隔多少个epoch进行一次验证')
    
    # ==================== Transformer架构参数 ====================
    # Transformer模型维度
    parser.add_argument('--d_model', type=int, default=256,
                       help='Transformer模型维度，用于transformer融合策略')
    
    # 多头注意力头数
    parser.add_argument('--n_heads', type=int, default=8,
                       help='多头注意力机制的头数，必须能被d_model整除')
    
    # 情绪类别数(重复参数，与num_classes相同)
    parser.add_argument('--num_emotions', type=int, default=7,
                       help='情绪类别数，与num_classes功能相同')
    
    return parser

def print_system_info():
    """打印系统信息""" 
    import torch
    
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    
    # CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"GPU {i}: {props.name} - {memory_gb:.1f}GB")
    else:
        print("CUDA不可用")
    
    # CPU信息
    print(f"CPU核心数: {os.cpu_count()}")
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    print("=" * 60)

def validate_gpu_setup(gpu_ids):
    """验证GPU设置"""
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行GPU训练")
    
    available_gpus = torch.cuda.device_count()
    
    if len(gpu_ids) > available_gpus:
        raise ValueError(f"指定的GPU数量({len(gpu_ids)})超过可用GPU数量({available_gpus})")
    
    for gpu_id in gpu_ids:
        if gpu_id >= available_gpus:
            raise ValueError(f"GPU {gpu_id}不存在，可用GPU: 0-{available_gpus-1}")
    
    # 检查GPU内存
    for gpu_id in gpu_ids:
        props = torch.cuda.get_device_properties(gpu_id)
        memory_gb = props.total_memory / 1024**3
        if memory_gb < 20:  # RTX 4090应该有24GB
            print(f"警告: GPU {gpu_id}内存较小({memory_gb:.1f}GB)，可能需要减小batch_size")

def optimize_environment():
    """优化训练环境"""
    # 设置环境变量以优化性能
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步CUDA操作
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # 启用cuDNN v8 API
    
    # 优化内存分配
    import torch
    if torch.cuda.is_available():
        # 启用内存池
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # 设置内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def main():
    """主函数"""
    # 打印系统信息
    print_system_info()
    
    # 解析参数
    parser = setup_optimal_args()
    args = parser.parse_args()
    
    # 验证GPU设置
    validate_gpu_setup(args.gpu_ids)
    
    # 优化环境
    optimize_environment()
    
    # 打印训练配置
    print("\n训练配置:")
    print(f"数据集: {args.dataset}")
    print(f"数据目录: {args.data_dir}")
    print(f"使用GPU: {args.gpu_ids}")
    print(f"每GPU批次大小: {args.batch_size}")
    print(f"总批次大小: {args.batch_size * len(args.gpu_ids)}")
    print(f"数据加载线程: {args.num_workers}")
    print(f"学习率 - Visual: {args.lr_visual:.2e}, Audio: {args.lr_audio:.2e}, Fusion: {args.lr_fusion:.2e}")
    print(f"融合类型: {args.fusion_type}")
    print(f"训练轮数: {args.epochs}")
    
    # 确认开始训练
    print("\n准备开始训练...")
    input("按Enter键继续，或Ctrl+C取消")
    
    # 调用原始训练函数
    try:
        # 导入训练模块
        import sys
        # 确保使用正确的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        multimodal_dir = os.path.join(current_dir, 'multimodal_emotion')
        sys.path.insert(0, multimodal_dir)
        
        # 明确从multimodal_emotion模块导入
        from multimodal_emotion.train import Trainer
        from multimodal_emotion.fusion_model import create_model
        from multimodal_emotion.data_utils import create_data_loaders
        
        # 创建数据加载器 - 优化多GPU性能
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True  # 启用内存固定以提高GPU传输效率
        )
        
        # 创建模型
        model = create_model(
            fusion_type=args.fusion_type,
            d_model=args.d_model,
            num_emotions=args.num_emotions
        )
        
        # 创建训练配置 - 包含多GPU设置
        config = {
            'num_epochs': args.epochs,
            'lr_visual': args.lr_visual,
            'lr_audio': args.lr_audio,
            'lr_fusion': args.lr_fusion,
            'weight_decay': args.weight_decay,
            'scheduler': args.scheduler,
            'patience': args.patience,
            'save_best_only': args.save_best_only,
            'early_stopping': args.early_stopping,
            'early_stopping_patience': args.early_stopping_patience,
            'gradient_clip': args.gradient_clip,
            'use_amp': args.use_amp,
            'log_interval': args.log_interval,
            'val_interval': args.val_interval,
            # 多GPU相关配置
            'use_multi_gpu': args.use_multi_gpu,
            'gpu_ids': args.gpu_ids,
            'optimizer': 'adam',  # 添加缺失的优化器配置
            'batch_size': args.batch_size,  # 添加batch_size到config中
            'scale_lr_with_gpu_count': True  # 启用学习率自动缩放
        }
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            save_dir=args.save_dir
        )
        
        # 开始训练
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()