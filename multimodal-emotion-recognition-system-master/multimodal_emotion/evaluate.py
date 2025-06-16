"""
评估和可视化工具
提供模型评估、结果可视化和分析功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Tuple
import cv2
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from fusion_model import create_model
from data_utils import create_data_loaders


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Args:
            model_path: 模型检查点路径
            config_path: 配置文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self._load_model(model_path, config_path)
        
        # 情绪标签
        self.emotion_labels = ['anger', 'disgust', 'sadness', 'joy', 
                              'neutral', 'surprise', 'fear']
        
        # 颜色映射
        self.emotion_colors = {
            'anger': '#FF6B6B',
            'disgust': '#95E1D3',
            'sadness': '#4ECDC4',
            'joy': '#FFE66D',
            'neutral': '#C9C9C9',
            'surprise': '#FF8B94',
            'fear': '#A8E6CF'
        }
    
    def _load_model(self, model_path: str, config_path: str = None):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = checkpoint.get('config', {})
        
        self.model = create_model(
            num_emotions=config.get('num_emotions', 7),
            d_model=config.get('d_model', 256),
            fusion_type=config.get('fusion_type', 'cross_attention')
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.config = config
    
    def evaluate_dataset(self, data_loader) -> Dict:
        """评估数据集"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for images, mel_specs, labels in data_loader:
                images = images.to(self.device)
                mel_specs = mel_specs.to(self.device)
                
                outputs = self.model(images, mel_specs)
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        accuracy = np.mean(all_preds == all_labels)
        report = classification_report(
            all_labels, all_preds,
            target_names=self.emotion_labels,
            output_dict=True
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: str = None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        
        # 归一化混淆矩阵
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # 创建热力图
        sns.heatmap(
            conf_matrix_norm,
            annot=conf_matrix,
            fmt='d',
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels,
            cbar_kws={'label': 'Proportion'},
            square=True
        )
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self, report: Dict, save_path: str = None):
        """绘制每类指标"""
        # 提取每类的精确率、召回率和F1分数
        metrics_data = []
        for label in self.emotion_labels:
            if label in report:
                metrics_data.append({
                    'Emotion': label.capitalize(),
                    'Precision': report[label]['precision'],
                    'Recall': report[label]['recall'],
                    'F1-Score': report[label]['f1-score']
                })
        
        df = pd.DataFrame(metrics_data)
        
        # 创建分组条形图
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.25
        
        bars1 = ax.bar(x - width, df['Precision'], width, label='Precision',
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x, df['Recall'], width, label='Recall',
                       color='#4ECDC4', alpha=0.8)
        bars3 = ax.bar(x + width, df['F1-Score'], width, label='F1-Score',
                       color='#FFE66D', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Emotion'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_emotion_distribution(self, labels: np.ndarray, predictions: np.ndarray,
                                 save_path: str = None):
        """绘制情绪分布对比"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 真实标签分布
        true_counts = np.bincount(labels, minlength=len(self.emotion_labels))
        ax1.pie(true_counts, labels=self.emotion_labels, autopct='%1.1f%%',
               colors=[self.emotion_colors[e] for e in self.emotion_labels])
        ax1.set_title('True Emotion Distribution', fontsize=14)
        
        # 预测标签分布
        pred_counts = np.bincount(predictions, minlength=len(self.emotion_labels))
        ax2.pie(pred_counts, labels=self.emotion_labels, autopct='%1.1f%%',
               colors=[self.emotion_colors[e] for e in self.emotion_labels])
        ax2.set_title('Predicted Emotion Distribution', fontsize=14)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_errors(self, data_loader, num_samples: int = 10) -> List[Dict]:
        """分析错误案例"""
        errors = []
        
        self.model.eval()
        with torch.no_grad():
            for images, mel_specs, labels in data_loader:
                images = images.to(self.device)
                mel_specs = mel_specs.to(self.device)
                
                outputs = self.model(images, mel_specs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                # 找出错误预测
                mask = preds != labels.to(self.device)
                if mask.any():
                    error_indices = torch.where(mask)[0]
                    
                    for idx in error_indices:
                        error_info = {
                            'true_label': self.emotion_labels[labels[idx].item()],
                            'pred_label': self.emotion_labels[preds[idx].item()],
                            'confidence': probs[idx, preds[idx]].item(),
                            'true_prob': probs[idx, labels[idx]].item(),
                            'all_probs': probs[idx].cpu().numpy(),
                            'image': images[idx].cpu(),
                            'mel_spec': mel_specs[idx].cpu()
                        }
                        errors.append(error_info)
                        
                        if len(errors) >= num_samples:
                            return errors
        
        return errors
    
    def visualize_attention_weights(self, image: torch.Tensor, mel_spec: torch.Tensor,
                                   save_path: str = None):
        """可视化注意力权重"""
        if not hasattr(self.model, 'get_attention_weights'):
            print("Model does not support attention visualization")
            return
        
        # 前向传播
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            mel_spec = mel_spec.unsqueeze(0).to(self.device)
            
            outputs = self.model(image, mel_spec)
            attention_weights = self.model.get_attention_weights()
        
        if attention_weights is None:
            print("No attention weights available")
            return
        
        # 可视化注意力权重
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # V2A注意力
        if 'v2a' in attention_weights and attention_weights['v2a'] is not None:
            attn_v2a = attention_weights['v2a'].squeeze().cpu().numpy()
            im1 = axes[0].imshow(attn_v2a, cmap='hot', interpolation='nearest')
            axes[0].set_title('Visual to Audio Attention')
            axes[0].set_xlabel('Audio Features')
            axes[0].set_ylabel('Visual Features')
            plt.colorbar(im1, ax=axes[0])
        
        # A2V注意力
        if 'a2v' in attention_weights and attention_weights['a2v'] is not None:
            attn_a2v = attention_weights['a2v'].squeeze().cpu().numpy()
            im2 = axes[1].imshow(attn_a2v, cmap='hot', interpolation='nearest')
            axes[1].set_title('Audio to Visual Attention')
            axes[1].set_xlabel('Visual Features')
            axes[1].set_ylabel('Audio Features')
            plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results: Dict, save_dir: str):
        """生成评估报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文本报告
        report_text = f"""
Multimodal Emotion Recognition Evaluation Report
================================================

Model Configuration:
{json.dumps(self.config, indent=2)}

Overall Accuracy: {results['accuracy']:.4f}

Classification Report:
{pd.DataFrame(results['classification_report']).transpose().to_string()}

Confusion Matrix:
{results['confusion_matrix']}
"""
        
        with open(save_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report_text)
        
        # 生成可视化
        self.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=str(save_dir / 'confusion_matrix.png')
        )
        
        self.plot_per_class_metrics(
            results['classification_report'],
            save_path=str(save_dir / 'per_class_metrics.png')
        )
        
        self.plot_emotion_distribution(
            results['labels'],
            results['predictions'],
            save_path=str(save_dir / 'emotion_distribution.png')
        )
        
        # 保存详细结果
        detailed_results = {
            'config': self.config,
            'accuracy': float(results['accuracy']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        with open(save_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Evaluation report saved to {save_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Multimodal Emotion Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str,
                       help='Path to model config')
    parser.add_argument('--dataset', type=str, default='MELD',
                       choices=['MELD', 'IEMOCAP'])
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, args.config_path)
    
    # 创建数据加载器
    _, _, test_loader = create_data_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # 评估
    print("Evaluating model on test set...")
    results = evaluator.evaluate_dataset(test_loader)
    
    # 生成报告
    evaluator.generate_report(results, args.save_dir)
    
    # 分析错误案例
    print("\nAnalyzing error cases...")
    errors = evaluator.analyze_errors(test_loader, num_samples=5)
    print(f"Found {len(errors)} error cases for analysis")


if __name__ == '__main__':
    main()
