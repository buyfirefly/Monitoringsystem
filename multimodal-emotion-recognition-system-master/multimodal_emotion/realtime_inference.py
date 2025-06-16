"""
实时推理接口
支持从摄像头和麦克风实时采集数据进行情绪识别
"""

import cv2
import torch
import torchaudio
import numpy as np
import pyaudio
import threading
import queue
import time
from pathlib import Path
import argparse
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from fusion_model import create_model
from data_utils import EmotionDatasetMELD


class RealTimeEmotionRecognizer:
    """实时情绪识别器"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = 'cpu',
        audio_duration: float = 3.0,
        update_interval: float = 0.5,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_path: 训练好的模型路径
            config_path: 配置文件路径
            device: 运行设备
            audio_duration: 音频片段长度（秒）
            update_interval: 更新间隔（秒）
            confidence_threshold: 置信度阈值
        """
        self.device = torch.device(device)
        self.audio_duration = audio_duration
        self.update_interval = update_interval
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        self._load_model(model_path, config_path)
        
        # 音频参数
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_buffer = deque(maxlen=int(self.sample_rate * audio_duration))
        self.audio_queue = queue.Queue()
        
        # 视频参数
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        
        # 情绪标签
        self.emotion_labels = ['anger', 'disgust', 'sadness', 'joy', 
                              'neutral', 'surprise', 'fear']
        
        # 情绪表情符号（用于显示）
        self.emotion_emojis = {
            'anger': '😠', 'disgust': '🤢', 'sadness': '😢',
            'joy': '😄', 'neutral': '😐', 'surprise': '😮', 'fear': '😨'
        }
        
        # 图像和音频变换
        self._setup_transforms()
        
        # 运行标志
        self.running = False
    
    def _load_model(self, model_path: str, config_path: str = None):
        """加载模型"""
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取配置
        if config_path:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = checkpoint.get('config', {})
        
        # 创建模型
        self.model = create_model(
            num_emotions=config.get('num_emotions', 7),
            d_model=config.get('d_model', 256),
            fusion_type=config.get('fusion_type', 'cross_attention')
        )
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def _setup_transforms(self):
        """设置图像和音频变换"""
        # 图像变换
        from torchvision import transforms
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 音频变换（梅尔频谱）
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            win_length=400
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数"""
        # 将音频数据放入队列
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def audio_thread(self):
        """音频采集线程"""
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        stream.start_stream()
        
        while self.running:
            try:
                # 从队列获取音频数据
                audio_data = self.audio_queue.get(timeout=0.1)
                # 转换为numpy数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                # 添加到缓冲区
                self.audio_buffer.extend(audio_array)
            except queue.Empty:
                continue
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def video_thread(self):
        """视频采集线程"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.frame_buffer = frame.copy()
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
    
    def process_inputs(self):
        """处理输入数据并进行推理"""
        # 获取当前帧
        with self.frame_lock:
            if self.frame_buffer is None:
                return None
            frame = self.frame_buffer.copy()
        
        # 获取音频数据
        if len(self.audio_buffer) < self.sample_rate * self.audio_duration:
            return None
        
        audio_data = np.array(list(self.audio_buffer))
        
        # 预处理图像
        # 检测人脸（可选）
        face_frame = self.detect_and_crop_face(frame)
        if face_frame is None:
            face_frame = frame  # 如果没有检测到人脸，使用整个帧
        
        # 转换图像
        img_tensor = self.img_transform(face_frame).unsqueeze(0).to(self.device)
        
        # 预处理音频
        # 归一化音频
        audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
        audio_tensor = audio_tensor.unsqueeze(0)  # 添加batch维度
        
        # 计算梅尔频谱
        mel_spec = self.mel_transform(audio_tensor)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = mel_spec.squeeze(0).to(self.device)  # (n_mels, time)
        mel_spec = mel_spec.unsqueeze(0)  # 添加batch维度
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(img_tensor, mel_spec)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = probs.max(1)
        
        # 获取结果
        emotion = self.emotion_labels[pred_idx.item()]
        confidence_score = confidence.item()
        
        # 只有当置信度超过阈值时才返回结果
        if confidence_score >= self.confidence_threshold:
            return {
                'emotion': emotion,
                'confidence': confidence_score,
                'probabilities': probs.squeeze().cpu().numpy(),
                'frame': frame
            }
        else:
            return None
    
    def detect_and_crop_face(self, frame):
        """检测并裁剪人脸"""
        # 使用OpenCV的人脸检测器
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # 选择最大的人脸
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # 扩展边界框
            padding = int(0.2 * w)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            face_frame = frame[y:y+h, x:x+w]
            # 转换为RGB
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            return face_frame
        
        return None
    
    def draw_results(self, frame, result):
        """在帧上绘制结果"""
        if result is None:
            return frame
        
        emotion = result['emotion']
        confidence = result['confidence']
        probs = result['probabilities']
        
        # 绘制主要情绪
        emoji = self.emotion_emojis.get(emotion, '')
        text = f"{emoji} {emotion.capitalize()}: {confidence:.2%}"
        cv2.putText(frame, text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # 绘制所有情绪的概率条
        bar_width = 200
        bar_height = 20
        start_y = 80
        
        for i, (label, prob) in enumerate(zip(self.emotion_labels, probs)):
            y = start_y + i * 30
            # 背景条
            cv2.rectangle(frame, (20, y), (20 + bar_width, y + bar_height),
                         (200, 200, 200), -1)
            # 概率条
            filled_width = int(bar_width * prob)
            color = (0, 255, 0) if label == emotion else (100, 100, 100)
            cv2.rectangle(frame, (20, y), (20 + filled_width, y + bar_height),
                         color, -1)
            # 标签
            cv2.putText(frame, f"{label}: {prob:.2%}", 
                       (25 + bar_width, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def run(self):
        """运行实时情绪识别"""
        self.running = True
        
        # 启动音频和视频线程
        audio_thread = threading.Thread(target=self.audio_thread)
        video_thread = threading.Thread(target=self.video_thread)
        
        audio_thread.start()
        video_thread.start()
        
        print("Starting real-time emotion recognition...")
        print("Press 'q' to quit")
        
        last_update = time.time()
        current_result = None
        
        try:
            while self.running:
                # 定期更新预测
                if time.time() - last_update >= self.update_interval:
                    result = self.process_inputs()
                    if result is not None:
                        current_result = result
                    last_update = time.time()
                
                # 显示结果
                with self.frame_lock:
                    if self.frame_buffer is not None:
                        display_frame = self.frame_buffer.copy()
                        display_frame = self.draw_results(display_frame, current_result)
                        cv2.imshow('Multimodal Emotion Recognition', display_frame)
                
                # 检查退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            audio_thread.join()
            video_thread.join()
            cv2.destroyAllWindows()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Real-time Multimodal Emotion Recognition')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str,
                       help='Path to model config (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--audio_duration', type=float, default=3.0,
                       help='Audio segment duration in seconds')
    parser.add_argument('--update_interval', type=float, default=0.5,
                       help='Prediction update interval in seconds')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # 创建实时识别器
    recognizer = RealTimeEmotionRecognizer(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        audio_duration=args.audio_duration,
        update_interval=args.update_interval,
        confidence_threshold=args.confidence_threshold
    )
    
    # 运行
    recognizer.run()


if __name__ == '__main__':
    main()
