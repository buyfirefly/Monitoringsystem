"""
数据预处理和加载模块
支持MELD和IEMOCAP数据集
"""

import os
import cv2
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, List, Dict
import subprocess
from pathlib import Path


class EmotionDatasetMELD(Dataset):
    """MELD数据集加载器"""
    
    # MELD情绪标签映射
    EMOTION_LABELS = {
        'anger': 0, 'disgust': 1, 'sadness': 2, 'joy': 3,
        'neutral': 4, 'surprise': 5, 'fear': 6
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform_img: Optional[transforms.Compose] = None,
        transform_audio: Optional[torchaudio.transforms.MelSpectrogram] = None,
        audio_length: float = 3.0,  # 音频长度（秒）
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_dir: MELD数据集根目录
            split: 'train', 'dev', 或 'test'
            transform_img: 图像变换
            transform_audio: 音频变换（梅尔频谱）
            audio_length: 统一的音频长度
            cache_dir: 缓存预处理数据的目录
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_length = audio_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 设置默认变换
        self.transform_img = transform_img or self._get_default_img_transform()
        self.transform_audio = transform_audio or self._get_default_audio_transform()
        
        # 加载CSV文件
        csv_file = self.data_dir / 'data' / 'MELD' / f'{split}_sent_emo.csv'
        self.df = pd.read_csv(csv_file)
        
        # 构建文件路径
        self.df['video_path'] = self.df.apply(
            lambda row: self.data_dir / f'{split}_splits' / 
            f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4",
            axis=1
        )
        
        # 创建缓存目录
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_features()
    
    def _get_default_img_transform(self):
        """默认图像变换"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_default_audio_transform(self):
        """默认音频变换（梅尔频谱）"""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            win_length=400
        )
    
    def _cache_features(self):
        """预先提取并缓存特征"""
        print(f"Caching features for {self.split} split...")
        # 这里可以实现批量特征提取和缓存逻辑
        pass
    
    def _get_default_frame(self) -> np.ndarray:
        """返回默认的黑色图像帧"""
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _extract_video_frame(self, video_path: str, frame_position: float = 0.5) -> np.ndarray:
        """从视频中提取帧，包含错误处理"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # 检查视频是否能正常打开
            if not cap.isOpened():
                print(f"Warning: Cannot open video {video_path}, using default frame")
                return self._get_default_frame()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 检查视频是否包含帧
            if total_frames <= 0:
                print(f"Warning: Video {video_path} has no frames, using default frame")
                cap.release()
                return self._get_default_frame()
            
            # 提取指定位置的帧
            target_frame = int(total_frames * frame_position)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Warning: Failed to read frame from {video_path}, using default frame")
                return self._get_default_frame()
            
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
            
        except Exception as e:
            print(f"Warning: Error extracting frame from {video_path}: {e}, using default frame")
            return self._get_default_frame()
    
    def _get_default_audio(self) -> torch.Tensor:
        """返回默认的静音音频"""
        target_length = int(self.audio_length * 16000)  # 16kHz采样率
        return torch.zeros(1, target_length)
    
    def _extract_audio(self, video_path: str) -> torch.Tensor:
        """从视频中提取音频，包含错误处理"""
        temp_audio = None
        try:
            # 临时音频文件
            temp_audio = str(video_path).replace('.mp4', '_temp.wav')
            
            # 使用ffmpeg提取音频
            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-ac', '1', '-ar', '16000', '-f', 'wav', temp_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            # 检查ffmpeg是否成功执行
            if result.returncode != 0:
                print(f"Warning: ffmpeg failed for {video_path}, using default audio")
                return self._get_default_audio()
            
            # 检查临时文件是否存在
            if not os.path.exists(temp_audio):
                print(f"Warning: Temp audio file not created for {video_path}, using default audio")
                return self._get_default_audio()
            
            # 加载音频
            waveform, sr = torchaudio.load(temp_audio)
            
            # 删除临时文件
            os.remove(temp_audio)
            
            # 调整音频长度
            target_length = int(self.audio_length * sr)
            if waveform.size(1) > target_length:
                # 截断
                waveform = waveform[:, :target_length]
            else:
                # 填充
                padding = target_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform
            
        except Exception as e:
            print(f"Warning: Error extracting audio from {video_path}: {e}, using default audio")
            # 清理临时文件
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except:
                    pass
            return self._get_default_audio()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            image: (3, H, W)
            mel_spec: (n_mels, time_frames)
            label: int
        """
        try:
            row = self.df.iloc[idx]
            video_path = row['video_path']
            
            # 提取图像帧
            frame = self._extract_video_frame(video_path)
            image = self.transform_img(frame)
            
            # 提取音频
            waveform = self._extract_audio(video_path)
            mel_spec = self.transform_audio(waveform)
            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            mel_spec = mel_spec.squeeze(0)  # 移除通道维度
            
            # 获取标签
            emotion = row['Emotion'].lower()
            label = self.EMOTION_LABELS[emotion]
            
            return image, mel_spec, label
            
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}, using default data")
            
            # 使用默认数据
            default_frame = self._get_default_frame()
            image = self.transform_img(default_frame)
            
            default_waveform = self._get_default_audio()
            mel_spec = self.transform_audio(default_waveform)
            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            mel_spec = mel_spec.squeeze(0)
            
            # 使用neutral作为默认标签
            label = self.EMOTION_LABELS['neutral']
            
            return image, mel_spec, label


class EmotionDatasetIEMOCAP(Dataset):
    """IEMOCAP数据集加载器"""
    
    # IEMOCAP情绪标签映射（4类）
    EMOTION_LABELS = {
        'ang': 0,  # anger
        'hap': 1,  # happiness (包括excitement)
        'exc': 1,  # excitement (映射到happiness)
        'sad': 2,  # sadness
        'neu': 3   # neutral
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sessions: List[int] = None,
        transform_img: Optional[transforms.Compose] = None,
        transform_audio: Optional[torchaudio.transforms.MelSpectrogram] = None,
        audio_length: float = 3.0
    ):
        """
        Args:
            data_dir: IEMOCAP数据集根目录
            split: 'train' 或 'test'
            sessions: 使用的session列表（1-5）
            transform_img: 图像变换
            transform_audio: 音频变换
            audio_length: 统一的音频长度
        """
        self.data_dir = Path(data_dir)
        self.audio_length = audio_length
        
        # 设置默认变换
        self.transform_img = transform_img or self._get_default_img_transform()
        self.transform_audio = transform_audio or self._get_default_audio_transform()
        
        # 设置sessions
        if sessions is None:
            # 默认train: session 1-4, test: session 5
            sessions = [1, 2, 3, 4] if split == 'train' else [5]
        
        # 加载数据列表
        self.data_list = self._load_data_list(sessions)
    
    def _get_default_img_transform(self):
        """默认图像变换"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_default_audio_transform(self):
        """默认音频变换"""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            win_length=400
        )
    
    def _load_data_list(self, sessions: List[int]) -> List[Dict]:
        """加载指定session的数据"""
        data_list = []
        
        for session in sessions:
            session_dir = self.data_dir / f'Session{session}'
            
            # 读取情绪标签文件
            label_file = session_dir / 'dialog' / 'EmoEvaluation' / '*.txt'
            
            # 遍历所有对话
            for label_path in session_dir.glob('dialog/EmoEvaluation/*.txt'):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.startswith('['):
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            utterance_id = parts[1]
                            emotion = parts[2]
                            
                            if emotion in self.EMOTION_LABELS:
                                # 构建音频和视频路径
                                dialog_id = utterance_id.split('_')[0]
                                
                                audio_path = session_dir / 'sentences' / 'wav' / \
                                           f'{dialog_id}' / f'{utterance_id}.wav'
                                
                                video_path = session_dir / 'dialog' / 'avi' / \
                                           f'{dialog_id}' / f'{utterance_id}.avi'
                                
                                if audio_path.exists() and video_path.exists():
                                    data_list.append({
                                        'utterance_id': utterance_id,
                                        'audio_path': audio_path,
                                        'video_path': video_path,
                                        'emotion': emotion,
                                        'label': self.EMOTION_LABELS[emotion]
                                    })
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """返回图像、音频频谱和标签"""
        data = self.data_list[idx]
        
        # 处理视频
        frame = self._extract_video_frame(str(data['video_path']))
        image = self.transform_img(frame)
        
        # 处理音频
        waveform, sr = torchaudio.load(data['audio_path'])
        
        # 重采样到16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # 调整长度
        target_length = int(self.audio_length * 16000)
        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]
        else:
            padding = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # 计算梅尔频谱
        mel_spec = self.transform_audio(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_spec = mel_spec.squeeze(0)
        
        return image, mel_spec, data['label']
    
    def _extract_video_frame(self, video_path: str) -> np.ndarray:
        """从视频提取中间帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 提取中间帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # 如果失败，创建黑色图像
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame


def create_data_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,  # 默认启用pin_memory以提高GPU传输效率
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器，针对多GPU训练进行优化
    
    Args:
        dataset_name: 'MELD' 或 'IEMOCAP'
        data_dir: 数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否启用内存固定（多GPU训练推荐True）
        **kwargs: 其他数据集特定参数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name.upper() == 'MELD':
        # 创建MELD数据集
        train_dataset = EmotionDatasetMELD(data_dir, split='train', **kwargs)
        val_dataset = EmotionDatasetMELD(data_dir, split='dev', **kwargs)
        test_dataset = EmotionDatasetMELD(data_dir, split='test', **kwargs)
        
    elif dataset_name.upper() == 'IEMOCAP':
        # 创建IEMOCAP数据集
        train_dataset = EmotionDatasetIEMOCAP(
            data_dir, split='train', sessions=[1, 2, 3, 4], **kwargs
        )
        val_dataset = EmotionDatasetIEMOCAP(
            data_dir, split='test', sessions=[5], **kwargs
        )
        test_dataset = val_dataset  # IEMOCAP通常只分train/test
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 创建数据加载器 - 多GPU优化设置
    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,  # 保持worker进程活跃
        'prefetch_factor': 2 if num_workers > 0 else 2,  # 预取因子
    }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader


# 工具函数
def collate_fn_pad(batch):
    """
    自定义collate函数，处理不同长度的音频
    """
    images, mel_specs, labels = zip(*batch)
    
    # 图像已经是固定大小，直接stack
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    # 对梅尔频谱进行padding到相同长度
    max_len = max(spec.size(1) for spec in mel_specs)
    padded_specs = []
    
    for spec in mel_specs:
        if spec.size(1) < max_len:
            padding = max_len - spec.size(1)
            spec = torch.nn.functional.pad(spec, (0, padding))
        padded_specs.append(spec)
    
    mel_specs = torch.stack(padded_specs)
    
    return images, mel_specs, labels
