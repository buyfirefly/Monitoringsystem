"""
Multimodal Emotion Recognition Model
融合EmoNet（视觉）和emotion2vec（音频）的多模态情绪识别模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, Optional
import sys
import os

# 添加父目录到路径以导入emonet和emotion2vec
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入现有的模型（静默模式）
try:
    from emonet.emonet.models.emonet import EmoNet
except ImportError:
    EmoNet = None

try:
    from emotion2vec.upstream.models.emotion2vec import Emotion2Vec
except ImportError:
    Emotion2Vec = None


class CrossModalAttention(nn.Module):
    """跨模态注意力机制模块"""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 多头注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key_value: (batch, seq_len_kv, d_model)
        Returns:
            output: (batch, seq_len_q, d_model)
        """
        # 跨模态注意力
        attn_output, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm1(query + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(query)
        output = self.norm2(query + self.dropout(ffn_output))
        
        return output


class VisualEncoder(nn.Module):
    """视觉编码器 - 基于ResNet提取面部表情特征"""
    def __init__(self, d_model: int = 256, pretrained: bool = True):
        super().__init__()
        
        # 使用预训练的ResNet18作为特征提取器
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        # 移除最后的分类层
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # 投影层：将ResNet特征映射到统一维度
        self.projection = nn.Sequential(
            nn.Linear(512, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 如果有预训练的EmoNet权重，可以在这里加载
        self.use_emonet = False
        if EmoNet is not None:
            try:
                # 尝试加载EmoNet预训练权重
                self.emonet = EmoNet(n_emotions=8)
                # 这里可以加载预训练权重
                # self.emonet.load_state_dict(torch.load('../emonet/pretrained/emonet_8.pth', weights_only=False))
                self.use_emonet = True
            except:
                self.use_emonet = False
        
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (batch, 3, H, W)
        Returns:
            features: (batch, d_model)
            sequence: (batch, 1, d_model) - 用于注意力机制
        """
        # 提取视觉特征
        features = self.feature_extractor(images)  # (batch, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch, 512)
        
        # 投影到统一维度
        features = self.projection(features)  # (batch, d_model)
        
        # 创建序列形式（长度为1）用于注意力机制
        sequence = features.unsqueeze(1)  # (batch, 1, d_model)
        
        return features, sequence


class AudioEncoder(nn.Module):
    """音频编码器 - 基于RNN/CNN提取语音情绪特征"""
    def __init__(self, n_mels: int = 64, d_model: int = 256):
        super().__init__()
        
        # CNN层处理梅尔频谱图
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # 保留时间维度
        )
        
        # RNN层处理时序信息
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 是否使用预训练的emotion2vec
        self.use_emotion2vec = False
        if Emotion2Vec is not None:
            try:
                # 这里可以初始化emotion2vec模型
                print("Emotion2Vec module found")
                self.use_emotion2vec = True
            except:
                self.use_emotion2vec = False
        
    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spec: (batch, n_mels, time_frames)
        Returns:
            features: (batch, d_model)
            sequence: (batch, seq_len, d_model)
        """
        # 添加通道维度
        x = mel_spec.unsqueeze(1)  # (batch, 1, n_mels, time)
        
        # CNN特征提取
        x = self.conv_layers(x)  # (batch, 128, 1, time')
        x = x.squeeze(2).transpose(1, 2)  # (batch, time', 128)
        
        # RNN处理时序
        rnn_out, hidden = self.rnn(x)  # rnn_out: (batch, time', d_model)
        
        # 使用最后的隐藏状态作为全局特征
        # hidden: (num_layers * num_directions, batch, hidden_size)
        hidden_fwd = hidden[-2]  # 前向最后层
        hidden_bwd = hidden[-1]  # 后向最后层
        features = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch, d_model)
        
        return features, rnn_out


class MultimodalEmotionRecognizer(nn.Module):
    """多模态情绪识别主模型"""
    def __init__(
        self,
        num_emotions: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        fusion_type: str = "cross_attention"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # 视觉和音频编码器
        self.visual_encoder = VisualEncoder(d_model=d_model)
        self.audio_encoder = AudioEncoder(d_model=d_model)
        
        # 融合模块
        if fusion_type == "cross_attention":
            # 双向跨模态注意力
            self.v2a_attention = CrossModalAttention(d_model, n_heads)
            self.a2v_attention = CrossModalAttention(d_model, n_heads)
            fusion_dim = d_model * 2
        elif fusion_type == "concat":
            # 简单拼接
            fusion_dim = d_model * 2
        elif fusion_type == "transformer":
            # 共享Transformer编码器
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=3
            )
            fusion_dim = d_model
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # 情绪分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )
        
    def forward(
        self,
        images: torch.Tensor,
        mel_specs: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, H, W)
            mel_specs: (batch, n_mels, time_frames)
            return_features: 是否返回中间特征
        Returns:
            logits: (batch, num_emotions)
            features: Optional[(visual_features, audio_features, fused_features)]
        """
        # 提取模态特征
        visual_features, visual_seq = self.visual_encoder(images)
        audio_features, audio_seq = self.audio_encoder(mel_specs)
        
        # 模态融合
        if self.fusion_type == "cross_attention":
            # 视觉特征对音频的注意力
            v2a_features = self.v2a_attention(visual_seq, audio_seq)
            v2a_features = v2a_features.mean(dim=1)  # (batch, d_model)
            
            # 音频特征对视觉的注意力
            a2v_features = self.a2v_attention(
                audio_seq.mean(dim=1, keepdim=True),  # 使用平均池化的音频特征
                visual_seq
            )
            a2v_features = a2v_features.squeeze(1)  # (batch, d_model)
            
            # 拼接融合特征
            fused_features = torch.cat([v2a_features, a2v_features], dim=1)
            
        elif self.fusion_type == "concat":
            # 简单拼接
            fused_features = torch.cat([visual_features, audio_features], dim=1)
            
        elif self.fusion_type == "transformer":
            # 准备序列：添加模态标识
            batch_size = visual_seq.size(0)
            
            # 创建模态位置编码
            visual_pos = torch.zeros_like(visual_seq)
            audio_pos = torch.ones_like(audio_seq) * 0.5
            
            # 拼接序列
            combined_seq = torch.cat([
                visual_seq + visual_pos,
                audio_seq + audio_pos
            ], dim=1)  # (batch, visual_len + audio_len, d_model)
            
            # Transformer编码
            encoded = self.transformer_encoder(combined_seq)
            
            # 使用[CLS]令牌或平均池化
            fused_features = encoded.mean(dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, (visual_features, audio_features, fused_features)
        return logits
    
    def get_attention_weights(self):
        """获取注意力权重用于可视化"""
        if hasattr(self, 'v2a_attention'):
            return {
                'v2a': self.v2a_attention.cross_attn.attn_weights,
                'a2v': self.a2v_attention.cross_attn.attn_weights
            }
        return None


def create_model(
    num_emotions: int = 7,
    d_model: int = 256,
    fusion_type: str = "cross_attention",
    pretrained: bool = True
) -> MultimodalEmotionRecognizer:
    """
    创建多模态情绪识别模型
    
    Args:
        num_emotions: 情绪类别数
        d_model: 特征维度
        fusion_type: 融合类型 ['cross_attention', 'concat', 'transformer']
        pretrained: 是否使用预训练权重
    """
    model = MultimodalEmotionRecognizer(
        num_emotions=num_emotions,
        d_model=d_model,
        fusion_type=fusion_type
    )
    return model


class MultiModalEmotionModel(nn.Module):
    """兼容旧版本模型的多模态情绪识别模型"""
    
    def __init__(self, visual_dim: int = 256, audio_dim: int = 256, 
                 hidden_dim: int = 512, num_classes: int = 7):
        super().__init__()
        
        # 视觉编码器 (使用ResNet18)
        self.visual_encoder = VisualEncoder(d_model=visual_dim)
        
        # 音频编码器 (简单的全连接层)
        self.audio_encoder = nn.Sequential(
            nn.Linear(64 * 500, audio_dim),  # 假设音频输入是64x500的梅尔频谱图
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 融合层 (输出256维以匹配分类器输入)
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 分类器 (根据实际checkpoint调整)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # 第0层: 256 -> 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # 第3层: 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # 最终分类层
        )
        
    def forward(self, visual_input, audio_input):
        # 视觉特征提取
        visual_features, _ = self.visual_encoder(visual_input)
        
        # 音频特征提取
        batch_size = audio_input.size(0)
        audio_flat = audio_input.reshape(batch_size, -1)
        audio_features = self.audio_encoder(audio_flat)
        
        # 特征融合
        combined_features = torch.cat([visual_features, audio_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        # 分类
        output = self.classifier(fused_features)
        
        return output
