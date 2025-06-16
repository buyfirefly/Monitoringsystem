#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时摄像头情绪监测
-------------------------------------------------
功能:
  1. 实时捕获摄像头视频流
  2. 检测人脸并提取面部表情
  3. 实时录制音频并提取语音特征
  4. 使用多模态模型进行情绪识别
  5. 实时显示情绪识别结果
  6. 支持结果保存和统计分析

使用示例:
  python realtime_camera_emotion.py --model_path ./checkpoints/best_model.pth
  python realtime_camera_emotion.py --model_path ./exported_models/model.onnx --save_results
"""

import cv2
import torch
import numpy as np
import pyaudio
import threading
import queue
import time
import argparse
from pathlib import Path
import json
from datetime import datetime
from collections import deque, Counter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
from typing import Optional, Tuple, Dict, List

# 导入模型相关模块
try:
    from multimodal_emotion.fusion_model import create_model
except ImportError:
    print("Warning: 无法导入fusion_model，请确保在正确的目录下运行")

try:
    import onnxruntime as ort
except ImportError:
    print("Warning: onnxruntime未安装，无法使用ONNX模型")
    ort = None


class AudioCapture:
    """音频捕获类"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def start_recording(self):
        """开始录音"""
        self.is_recording = True
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        print("🎤 音频录制已开始")
        
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("🎤 音频录制已停止")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
        
    def get_audio_features(self, duration=3.0):
        """获取音频特征"""
        # 收集指定时长的音频数据
        target_samples = int(self.sample_rate * duration)
        audio_buffer = []
        
        while len(audio_buffer) < target_samples:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.extend(chunk)
            except queue.Empty:
                break
                
        if len(audio_buffer) < target_samples:
            # 如果音频不够，用零填充
            audio_buffer.extend([0.0] * (target_samples - len(audio_buffer)))
        else:
            # 如果音频过多，截取
            audio_buffer = audio_buffer[:target_samples]
            
        audio_array = np.array(audio_buffer, dtype=np.float32)
        
        # 提取梅尔频谱图特征
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
            sr=self.sample_rate,
            n_mels=64,
            hop_length=512,
            win_length=1024
        )
        
        # 转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 调整到固定大小 (64, 500)
        if log_mel_spec.shape[1] < 500:
            # 如果时间维度不够，重复填充
            repeat_times = 500 // log_mel_spec.shape[1] + 1
            log_mel_spec = np.tile(log_mel_spec, (1, repeat_times))
        
        log_mel_spec = log_mel_spec[:, :500]  # 截取到500帧
        
        return log_mel_spec


class EmotionDetector:
    """情绪检测器"""
    
    def __init__(self, model_path: str, model_format: str = 'auto'):
        self.model_path = Path(model_path)
        self.model_format = model_format if model_format != 'auto' else self._detect_format()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 情绪标签
        self.emotion_labels = [
            'neutral', 'happiness', 'sadness', 'anger',
            'fear', 'disgust', 'surprise'
        ]
        
        # 情绪颜色映射（用于可视化）
        self.emotion_colors = {
            'neutral': (128, 128, 128),     # 灰色
            'happiness': (0, 255, 0),       # 绿色
            'sadness': (255, 0, 0),         # 蓝色
            'anger': (0, 0, 255),           # 红色
            'fear': (128, 0, 128),          # 紫色
            'disgust': (0, 128, 0),         # 深绿色
            'surprise': (255, 255, 0)       # 黄色
        }
        
        # 加载模型
        self.model = self._load_model()
        print(f"✅ 模型加载成功: {self.model_format} 格式")
        
        # 人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
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
        """加载模型"""
        if self.model_format == 'onnx':
            return self._load_onnx_model()
        elif self.model_format == 'torchscript':
            return self._load_torchscript_model()
        elif self.model_format == 'pytorch':
            return self._load_pytorch_model()
        else:
            raise ValueError(f"不支持的模型格式: {self.model_format}")
            
    def _load_onnx_model(self):
        """加载ONNX模型"""
        if ort is None:
            raise ImportError("请安装onnxruntime: pip install onnxruntime")
            
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
            
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        return session
        
    def _load_torchscript_model(self):
        """加载TorchScript模型"""
        model = torch.jit.load(self.model_path, map_location=self.device)
        model.eval()
        return model
        
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 检查模型结构
        state_dict = checkpoint['model_state_dict']
        
        # 如果模型包含classifier层，说明是旧版本的模型结构
        if any('classifier' in key for key in state_dict.keys()):
            print("检测到旧版本模型结构，使用兼容模式加载...")
            # 直接从state_dict重建模型
            from multimodal_emotion.fusion_model import MultiModalEmotionModel
            
            # 根据state_dict推断模型参数
            visual_dim = 256  # 从classifier层推断
            audio_dim = 256
            hidden_dim = 512
            num_classes = 7
            
            model = MultiModalEmotionModel(
                visual_dim=visual_dim,
                audio_dim=audio_dim, 
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
            
            # 加载权重，忽略不匹配的层
            model.load_state_dict(state_dict, strict=False)
        else:
            # 新版本模型结构
            config = checkpoint.get('config', {})
            model = create_model(
                num_emotions=config.get('num_classes', 7),
                d_model=config.get('d_model', 256),
                fusion_type=config.get('fusion_type', 'cross_attention'),
                pretrained=config.get('pretrained', True)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        return model
        
    def detect_faces(self, frame):
        """检测人脸"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
        
    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        # 调整大小到224x224
        face_resized = cv2.resize(face_img, (224, 224))
        
        # 转换为RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # ImageNet标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        
        # 转换为CHW格式并添加batch维度
        face_tensor = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_tensor, axis=0)
        
        return face_batch
        
    def predict_emotion(self, visual_input, audio_input):
        """预测情绪"""
        if self.model_format == 'onnx':
            # ONNX推理
            inputs = {
                'visual_input': visual_input.astype(np.float32),
                'audio_input': audio_input.astype(np.float32)
            }
            outputs = self.model.run(None, inputs)
            logits = outputs[0].squeeze()
        else:
            # PyTorch/TorchScript推理
            visual_tensor = torch.from_numpy(visual_input).float().to(self.device)
            audio_tensor = torch.from_numpy(audio_input).float().to(self.device)
            
            with torch.no_grad():
                logits = self.model(visual_tensor, audio_tensor)
                logits = logits.cpu().numpy().squeeze()
                
        # 计算概率
        probabilities = self._softmax(logits)
        predicted_class = np.argmax(probabilities)
        predicted_emotion = self.emotion_labels[predicted_class]
        confidence = probabilities[predicted_class]
        
        return predicted_emotion, confidence, probabilities
        
    def _softmax(self, x):
        """计算softmax概率"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class RealtimeEmotionMonitor:
    """实时情绪监测器"""
    
    def __init__(self, model_path: str, save_results: bool = False, 
                 show_stats: bool = True, camera_id: int = 0):
        self.detector = EmotionDetector(model_path)
        self.audio_capture = AudioCapture()
        self.save_results = save_results
        self.show_stats = show_stats
        self.camera_id = camera_id
        
        # 结果存储
        self.emotion_history = deque(maxlen=100)  # 保存最近100次检测结果
        self.results_log = []
        
        # 统计信息
        self.frame_count = 0
        self.start_time = time.time()
        
        # 可视化设置
        if self.show_stats:
            self.setup_visualization()
            
    def setup_visualization(self):
        """设置可视化界面"""
        plt.ion()  # 开启交互模式
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 情绪分布饼图
        self.ax1.set_title('情绪分布统计')
        
        # 情绪时间序列
        self.ax2.set_title('情绪变化趋势')
        self.ax2.set_xlabel('时间')
        self.ax2.set_ylabel('置信度')
        
    def update_visualization(self):
        """更新可视化界面"""
        if not self.show_stats or len(self.emotion_history) < 5:
            return
            
        # 清除之前的图
        self.ax1.clear()
        self.ax2.clear()
        
        # 情绪分布统计
        emotions = [result['emotion'] for result in self.emotion_history]
        emotion_counts = Counter(emotions)
        
        if emotion_counts:
            labels = list(emotion_counts.keys())
            sizes = list(emotion_counts.values())
            colors = [self.detector.emotion_colors.get(emotion, (128, 128, 128)) for emotion in labels]
            colors = [(r/255, g/255, b/255) for r, g, b in colors]  # 归一化颜色
            
            self.ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            self.ax1.set_title('情绪分布统计')
        
        # 情绪时间序列
        if len(self.emotion_history) > 1:
            times = [result['timestamp'] for result in self.emotion_history]
            confidences = [result['confidence'] for result in self.emotion_history]
            emotions = [result['emotion'] for result in self.emotion_history]
            
            # 为不同情绪使用不同颜色
            unique_emotions = list(set(emotions))
            for emotion in unique_emotions:
                emotion_times = [t for t, e in zip(times, emotions) if e == emotion]
                emotion_confidences = [c for c, e in zip(confidences, emotions) if e == emotion]
                color = self.detector.emotion_colors.get(emotion, (128, 128, 128))
                color = (color[0]/255, color[1]/255, color[2]/255)
                
                self.ax2.scatter(emotion_times, emotion_confidences, 
                               c=[color], label=emotion, alpha=0.7)
            
            self.ax2.set_title('情绪变化趋势')
            self.ax2.set_xlabel('时间 (秒)')
            self.ax2.set_ylabel('置信度')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)  # 短暂暂停以更新图形
        
    def run(self):
        """运行实时监测"""
        print("🚀 启动实时情绪监测...")
        print("按 'q' 键退出，按 's' 键保存当前结果")
        
        # 初始化摄像头
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区延迟
        
        # 开始音频录制
        self.audio_capture.start_recording()
        
        try:
            last_prediction_time = 0
            prediction_interval = 0.2  # 每200ms进行一次情绪预测
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                    
                self.frame_count += 1
                current_time = time.time() - self.start_time
                
                # 检测人脸
                faces = self.detector.detect_faces(frame)
                
                if len(faces) > 0:
                    # 使用最大的人脸
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    # 只在指定间隔内进行情绪预测
                    if current_time - last_prediction_time >= prediction_interval:
                        # 提取人脸区域
                        face_img = frame[y:y+h, x:x+w]
                        
                        # 预处理人脸图像
                        visual_input = self.detector.preprocess_face(face_img)
                        
                        # 获取音频特征
                        audio_input = self.audio_capture.get_audio_features()
                        audio_input = np.expand_dims(audio_input, axis=0)  # 添加batch维度
                        
                        # 进行情绪预测
                        emotion, confidence, probabilities = self.detector.predict_emotion(
                            visual_input, audio_input
                        )
                        
                        # 记录结果
                        result = {
                            'timestamp': current_time,
                            'emotion': emotion,
                            'confidence': confidence,
                            'probabilities': probabilities.tolist(),
                            'face_bbox': [x, y, w, h]
                        }
                        
                        self.emotion_history.append(result)
                        if self.save_results:
                            self.results_log.append(result)
                        
                        last_prediction_time = current_time
                    
                    # 使用最新的预测结果进行显示
                    if len(self.emotion_history) > 0:
                        latest_result = self.emotion_history[-1]
                        emotion = latest_result['emotion']
                        confidence = latest_result['confidence']
                        probabilities = np.array(latest_result['probabilities'])
                    
                    # 在画面上绘制结果
                    color = self.detector.emotion_colors.get(emotion, (128, 128, 128))
                    
                    # 绘制人脸框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # 绘制情绪标签
                    label = f"{emotion}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x, y-label_size[1]-10), 
                                (x+label_size[0], y), color, -1)
                    cv2.putText(frame, label, (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 绘制情绪概率条
                    bar_width = 200
                    bar_height = 15
                    start_y = 30
                    
                    for i, (emotion_name, prob) in enumerate(zip(self.detector.emotion_labels, probabilities)):
                        y_pos = start_y + i * (bar_height + 5)
                        bar_length = int(bar_width * prob)
                        
                        # 背景条
                        cv2.rectangle(frame, (10, y_pos), (10 + bar_width, y_pos + bar_height), 
                                    (50, 50, 50), -1)
                        
                        # 概率条
                        emotion_color = self.detector.emotion_colors.get(emotion_name, (128, 128, 128))
                        cv2.rectangle(frame, (10, y_pos), (10 + bar_length, y_pos + bar_height), 
                                    emotion_color, -1)
                        
                        # 标签
                        cv2.putText(frame, f"{emotion_name}: {prob:.2f}", 
                                  (10 + bar_width + 10, y_pos + bar_height - 3),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 显示FPS和统计信息
                fps = self.frame_count / current_time if current_time > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(self.emotion_history) > 0:
                    recent_emotion = self.emotion_history[-1]['emotion']
                    cv2.putText(frame, f"Current: {recent_emotion}", (frame.shape[1] - 150, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 显示画面
                cv2.imshow('实时情绪监测', frame)
                
                # 更新可视化
                if self.frame_count % 60 == 0:  # 每60帧更新一次统计图
                    self.update_visualization()
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_current_results()
                    
        except KeyboardInterrupt:
            print("\n⏹️  用户中断")
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            self.audio_capture.stop_recording()
            
            if self.show_stats:
                plt.ioff()
                plt.close()
            
            # 保存最终结果
            if self.save_results and self.results_log:
                self.save_final_results()
                
            print("✅ 监测结束")
            
    def save_current_results(self):
        """保存当前结果"""
        if not self.emotion_history:
            print("⚠️  没有可保存的结果")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_snapshot_{timestamp}.json"
        
        snapshot_data = {
            'timestamp': timestamp,
            'total_detections': len(self.emotion_history),
            'recent_emotions': list(self.emotion_history),
            'emotion_distribution': dict(Counter([r['emotion'] for r in self.emotion_history]))
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
        print(f"📸 当前结果已保存: {filename}")
        
    def save_final_results(self):
        """保存最终结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_session_{timestamp}.json"
        
        session_data = {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - self.start_time,
                'total_frames': self.frame_count,
                'total_detections': len(self.results_log)
            },
            'emotion_statistics': {
                'distribution': dict(Counter([r['emotion'] for r in self.results_log])),
                'average_confidence': np.mean([r['confidence'] for r in self.results_log]),
                'dominant_emotion': Counter([r['emotion'] for r in self.results_log]).most_common(1)[0][0]
            },
            'detailed_results': self.results_log
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 完整会话结果已保存: {filename}")
        
        # 生成统计报告
        self.generate_report(session_data)
        
    def generate_report(self, session_data):
        """生成统计报告"""
        print("\n📊 会话统计报告:")
        print(f"   持续时间: {session_data['session_info']['duration_seconds']:.1f} 秒")
        print(f"   总帧数: {session_data['session_info']['total_frames']}")
        print(f"   检测次数: {session_data['session_info']['total_detections']}")
        print(f"   平均置信度: {session_data['emotion_statistics']['average_confidence']:.3f}")
        print(f"   主导情绪: {session_data['emotion_statistics']['dominant_emotion']}")
        
        print("\n🎭 情绪分布:")
        for emotion, count in session_data['emotion_statistics']['distribution'].items():
            percentage = count / session_data['session_info']['total_detections'] * 100
            print(f"   {emotion:>10}: {count:>3} 次 ({percentage:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='实时摄像头情绪监测')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='摄像头ID (默认: 0)')
    parser.add_argument('--save_results', action='store_true',
                       help='保存检测结果到文件')
    parser.add_argument('--no_stats', action='store_true',
                       help='不显示统计图表')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model_path).exists():
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
        
    # 检查依赖
    try:
        import cv2
        import pyaudio
        import librosa
    except ImportError as e:
        print(f"❌ 缺少必要依赖: {e}")
        print("请安装: pip install opencv-python pyaudio librosa")
        return
    
    try:
        # 创建监测器
        monitor = RealtimeEmotionMonitor(
            model_path=args.model_path,
            save_results=args.save_results,
            show_stats=not args.no_stats,
            camera_id=args.camera_id
        )
        
        # 开始监测
        monitor.run()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()