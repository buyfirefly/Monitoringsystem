"""
训练脚本
支持多模态情绪识别模型的训练、验证和测试
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from fusion_model import create_model
from data_utils import create_data_loaders


class Trainer:
    """训练器类"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: dict,
        save_dir: str = "./checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 多GPU设置
        self.use_multi_gpu = config.get('use_multi_gpu', True)
        self.gpu_ids = config.get('gpu_ids', None)
        
        # 设备设置
        if torch.cuda.is_available():
            if self.use_multi_gpu and torch.cuda.device_count() > 1:
                if self.gpu_ids:
                    # 使用指定的GPU
                    self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
                    print(f"Using GPUs: {self.gpu_ids}")
                else:
                    # 使用所有可用GPU
                    self.device = torch.device('cuda:0')
                    self.gpu_ids = list(range(torch.cuda.device_count()))
                    print(f"Using all available GPUs: {self.gpu_ids}")
                
                # 将模型移到主GPU
                self.model.to(self.device)
                
                # 使用DataParallel进行多GPU并行
                if len(self.gpu_ids) > 1:
                    self.model = DataParallel(self.model, device_ids=self.gpu_ids)
                    print(f"Model wrapped with DataParallel on {len(self.gpu_ids)} GPUs")
                    
                    # 调整batch size以充分利用GPU
                    effective_batch_size = config.get('batch_size', 32) * len(self.gpu_ids)
                    print(f"Effective batch size: {effective_batch_size} (base: {config.get('batch_size', 32)} x {len(self.gpu_ids)} GPUs)")
            else:
                self.device = torch.device('cuda:0')
                self.model.to(self.device)
                print("Using single GPU")
        else:
            self.device = torch.device('cpu')
            self.model.to(self.device)
            print("Using CPU (CUDA not available)")
            
        # 打印GPU内存信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {gpu_memory:.1f}GB")
        
        # 优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))
        
        # 保存配置
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    def _create_optimizer(self):
        """创建优化器，支持不同学习率，并根据GPU数量自动缩放"""
        # 获取参数组
        visual_params = []
        audio_params = []
        fusion_params = []
        
        for name, param in self.model.named_parameters():
            if 'visual_encoder' in name:
                visual_params.append(param)
            elif 'audio_encoder' in name:
                audio_params.append(param)
            else:
                fusion_params.append(param)
        
        # 根据GPU数量自动缩放学习率
        gpu_count = len(self.gpu_ids) if self.use_multi_gpu and self.gpu_ids else 1
        lr_scale = gpu_count if self.config.get('scale_lr_with_gpu_count', True) else 1
        
        if lr_scale > 1:
            print(f"Scaling learning rates by factor of {lr_scale} for {gpu_count} GPUs")
        
        # 设置不同的学习率
        param_groups = [
            {'params': visual_params, 'lr': self.config['lr_visual'] * lr_scale},
            {'params': audio_params, 'lr': self.config['lr_audio'] * lr_scale},
            {'params': fusion_params, 'lr': self.config['lr_fusion'] * lr_scale}
        ]
        
        # 创建优化器
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                param_groups,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config['scheduler'] == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.config['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=1e-6
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (images, mel_specs, labels) in enumerate(pbar):
            # 移动到设备 - 非阻塞传输以提高效率
            images = images.to(self.device, non_blocking=True)
            mel_specs = mel_specs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, mel_specs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'gpu_mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
            })
            
            # TensorBoard记录
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/Accuracy', 100.*correct/total, global_step)
                
                # 记录GPU内存使用
                if torch.cuda.is_available():
                    for i, gpu_id in enumerate(self.gpu_ids if hasattr(self, 'gpu_ids') and self.gpu_ids else [0]):
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        self.writer.add_scalar(f'GPU_{gpu_id}/Memory_Allocated_GB', memory_allocated, global_step)
                        self.writer.add_scalar(f'GPU_{gpu_id}/Memory_Reserved_GB', memory_reserved, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, loader=None, phase='Val'):
        """验证模型"""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, mel_specs, labels in tqdm(loader, desc=f"{phase}"):
                images = images.to(self.device, non_blocking=True)
                mel_specs = mel_specs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(images, mel_specs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        
        # 计算详细指标
        report = classification_report(
            all_labels, all_preds,
            output_dict=True,
            zero_division=0
        )
        
        return avg_loss, accuracy, report, all_labels, all_preds
    
    def train(self):
        """完整训练流程"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_report, _, _ = self.validate()
            
            # 记录到TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # Early stopping
            if self.early_stop_counter >= self.config.get('early_stop_patience', 10):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # 训练结束，在测试集上评估
        self.test()
        
        self.writer.close()
    
    def test(self):
        """在测试集上评估"""
        print("\nEvaluating on test set...")
        
        # 加载最佳模型
        self.load_checkpoint('best_model.pth')
        
        # 测试
        test_loss, test_acc, test_report, all_labels, all_preds = self.validate(
            self.test_loader, phase='Test'
        )
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
        
        # 保存混淆矩阵
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # 保存结果
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': test_report,
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        # 获取类别名称
        if hasattr(self.train_loader.dataset, 'EMOTION_LABELS'):
            labels = list(self.train_loader.dataset.EMOTION_LABELS.keys())
        else:
            labels = [str(i) for i in range(self.config['num_emotions'])]
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        torch.save(checkpoint, self.save_dir / filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded: {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train Multimodal Emotion Recognition Model')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='MELD', choices=['MELD', 'IEMOCAP'])
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers per GPU')
    
    # 多GPU参数
    parser.add_argument('--use_multi_gpu', action='store_true', default=True, help='Use multiple GPUs')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2], help='GPU IDs to use (default: [0, 1, 2] for 3x4090)')
    parser.add_argument('--auto_scale_lr', action='store_true', default=True, help='Auto scale learning rate with number of GPUs')
    
    # 模型参数
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                       choices=['cross_attention', 'concat', 'transformer'])
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_emotions', type=int, default=7)
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr_visual', type=float, default=1e-4)
    parser.add_argument('--lr_audio', type=float, default=1e-4)
    parser.add_argument('--lr_fusion', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                       choices=['reduce_on_plateau', 'cosine', 'none'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 根据数据集调整参数
    if args.dataset == 'IEMOCAP':
        args.num_emotions = 4  # IEMOCAP只有4类情绪
    
    # 创建配置字典
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 多GPU设置
    if args.use_multi_gpu and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if args.gpu_ids:
            # 验证指定的GPU是否可用
            valid_gpu_ids = [gpu_id for gpu_id in args.gpu_ids if gpu_id < available_gpus]
            if len(valid_gpu_ids) != len(args.gpu_ids):
                print(f"Warning: Some specified GPUs are not available. Using: {valid_gpu_ids}")
            args.gpu_ids = valid_gpu_ids
        else:
            args.gpu_ids = list(range(available_gpus))
        
        num_gpus = len(args.gpu_ids)
        print(f"Using {num_gpus} GPUs: {args.gpu_ids}")
        
        # 自动调整学习率
        if args.auto_scale_lr and num_gpus > 1:
            scale_factor = num_gpus
            args.lr_visual *= scale_factor
            args.lr_audio *= scale_factor
            args.lr_fusion *= scale_factor
            print(f"Auto-scaled learning rates by {scale_factor}x for {num_gpus} GPUs")
        
        # 调整num_workers以充分利用CPU
        args.num_workers = min(args.num_workers * num_gpus, os.cpu_count())
        print(f"Using {args.num_workers} data loading workers")
    else:
        num_gpus = 1
        args.gpu_ids = [0] if torch.cuda.is_available() else None
    
    # 创建数据加载器
    print(f"Loading {args.dataset} dataset from {args.data_dir}")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()  # 启用pin_memory以提高GPU传输效率
    )
    
    print(f"Effective batch size: {args.batch_size * num_gpus} (base: {args.batch_size} x {num_gpus} GPUs)")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 创建模型
    print(f"Creating model with fusion type: {args.fusion_type}")
    model = create_model(
        num_emotions=args.num_emotions,
        d_model=args.d_model,
        fusion_type=args.fusion_type
    )
    
    # 创建训练器
    save_dir = Path(args.save_dir) / f"{args.dataset}_{args.fusion_type}_{config['timestamp']}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        save_dir=str(save_dir)
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
