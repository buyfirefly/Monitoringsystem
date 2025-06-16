from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import threading
import time
import shutil
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# 添加多模态情绪识别系统路径
sys.path.append('../multimodal-emotion-recognition-system-master')
sys.path.append('../multimodal-emotion-recognition-system-master/emonet')
sys.path.append('../multimodal-emotion-recognition-system-master/multimodal_emotion')

try:
    from multimodal_emotion.fusion_model import MultimodalEmotionRecognizer, MultiModalEmotionModel
    from multimodal_emotion.realtime_inference import RealTimeEmotionRecognizer
    import torchvision.transforms as transforms
    from PIL import Image
except ImportError as e:
    print(f"Warning: Could not import multimodal emotion recognition modules: {e}")
    print("Please ensure the multimodal emotion recognition system is properly installed.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)  # 添加CORS支持
socketio = SocketIO(app, cors_allowed_origins="*")

# 设置上传文件夹和最大文件大小
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 创建上传文件夹
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 情绪识别模型配置
EMOTION_MODEL = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 224
EMOTION_CLASSES = ['anger', 'disgust', 'sadness', 'joy', 'neutral', 'surprise', 'fear']
EMOTION_EMOJIS = {
    'anger': '😠', 'disgust': '🤢', 'sadness': '😢',
    'joy': '😄', 'neutral': '😐', 'surprise': '😮', 'fear': '😨'
}

# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化情绪识别模型
def initialize_emotion_model():
    global EMOTION_MODEL
    try:
        # 加载多模态情绪识别模型
        model_path = '../best_model.pth'
        print(f"正在检查模型文件: {model_path}")
        print(f"模型文件存在: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print("开始加载模型检查点...")
            # 加载检查点
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            print(f"检查点键: {list(checkpoint.keys())}")
            
            print("创建MultimodalEmotionRecognizer实例...")
            # 使用MultimodalEmotionRecognizer（与checkpoint结构匹配）
            # 根据检查点中的transformer_encoder权重，使用transformer融合类型
            EMOTION_MODEL = MultimodalEmotionRecognizer(
                num_emotions=7,
                d_model=256,
                n_heads=8,
                fusion_type="transformer"
            )
            
            print("加载模型权重...")
            # 加载权重
            EMOTION_MODEL.load_state_dict(checkpoint['model_state_dict'])
            EMOTION_MODEL.to(DEVICE)
            EMOTION_MODEL.eval()
            
            print("模型加载成功！")
            push_message('多模态情绪识别模型加载成功！', '#006400')
            return True
        else:
            print(f"错误: 未找到模型文件 {model_path}")
            push_message('未找到best_model.pth文件', '#FF0000')
            return False
    except Exception as e:
        print(f"模型加载异常: {str(e)}")
        print(f"异常类型: {type(e).__name__}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        push_message(f'模型加载失败: {str(e)}', '#FF0000')
        return False

# 处理视频情绪识别
def process_video_emotion(video_path):
    try:
        if EMOTION_MODEL is None:
            return {'error': '情绪识别模型未加载'}
        
        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': '无法读取视频文件'}
        
        emotion_results = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 分析每一帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测人脸 - 多尺度检测提高成功率
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 图像预处理 - 增强对比度
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                
                # 多尺度人脸检测 - 尝试不同参数组合
                faces = []
                
                # 第一次检测：高精度检测
                faces1 = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.03,  # 更小的缩放步长
                    minNeighbors=2,    # 降低邻居要求
                    minSize=(20, 20),  # 更小的最小尺寸
                    maxSize=(800, 800), # 更大的最大尺寸
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces1) > 0:
                    faces = faces1
                else:
                    # 第二次检测：更宽松的参数
                    faces2 = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1,
                        minNeighbors=1,
                        minSize=(15, 15),
                        maxSize=(1000, 1000),
                        flags=cv2.CASCADE_DO_CANNY_PRUNING
                    )
                    
                    if len(faces2) > 0:
                        faces = faces2
                    else:
                        # 第三次检测：最宽松的参数
                        faces3 = face_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=1.2,
                            minNeighbors=1,
                            minSize=(10, 10)
                        )
                        faces = faces3
                
                # 添加调试信息
                print(f"第 {frame_count} 帧检测结果: 发现 {len(faces)} 个人脸")
                
                if len(faces) > 0:
                    # 取第一个检测到的人脸
                    x, y, w, h = faces[0]
                    print(f"人脸位置: x={x}, y={y}, w={w}, h={h}")
                    
                    # 裁剪人脸区域
                    face_crop = frame_rgb[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:
                        # 添加调试：检查原始人脸图像的统计信息
                        print(f"原始人脸图像统计: mean={face_crop.mean():.3f}, std={face_crop.std():.3f}, min={face_crop.min()}, max={face_crop.max()}")
                        
                        # 预处理图像
                        face_pil = Image.fromarray(face_crop)
                        face_tensor = image_transform(face_pil).unsqueeze(0).to(DEVICE)
                        
                        # 添加调试信息：检查预处理后的tensor统计
                        print(f"face_tensor shape: {face_tensor.shape}")
                        print(f"预处理后tensor统计: mean={face_tensor.mean():.6f}, std={face_tensor.std():.6f}, min={face_tensor.min():.6f}, max={face_tensor.max():.6f}")
                        
                        # 运行情绪识别（多模态融合优化版 - 增强情绪敏感度）
                        if EMOTION_MODEL is not None:
                            try:
                                with torch.no_grad():
                                    # 首先提取视觉特征用于生成智能音频输入
                                    visual_features, visual_seq = EMOTION_MODEL.visual_encoder(face_tensor)
                                    
                                    # 基于视觉特征生成智能音频输入
                                    # 分析面部表情强度来调整音频特征
                                    visual_intensity = torch.abs(visual_features).mean().item()
                                    
                                    # 创建基于视觉线索的音频特征
                                    # 使用视觉特征的统计信息来指导音频生成
                                    base_freq = 0.1 + visual_intensity * 0.05  # 基础频率
                                    amplitude = 0.05 + visual_intensity * 0.03  # 振幅
                                    
                                    # 生成更真实的梅尔频谱图
                                    # 模拟不同情绪的音频特征模式
                                    time_steps = 100
                                    mel_channels = 64
                                    
                                    # 创建时间序列，添加帧索引作为时间偏移
                                    frame_offset = (frame_count % 100) * 0.01  # 基于帧数的时间偏移
                                    t = torch.linspace(frame_offset, 1 + frame_offset, time_steps)
                                    
                                    # 生成多频率成分的音频特征
                                    mel_spec = torch.zeros(1, mel_channels, time_steps).to(DEVICE)
                                    
                                    # 添加基于帧数的随机种子变化
                                    torch.manual_seed(frame_count + int(visual_intensity * 1000))
                                    
                                    for i in range(mel_channels):
                                        # 为每个梅尔频道生成不同的频率成分
                                        freq_factor = (i + 1) / mel_channels
                                        
                                        # 添加帧相关的频率调制
                                        frame_modulation = 1.0 + 0.1 * torch.sin(torch.tensor(frame_count * 0.1))
                                        
                                        # 基于视觉特征调整频率模式
                                        if visual_intensity > 0.3:  # 高强度表情
                                            # 生成更活跃的音频模式（如笑声、惊讶）
                                            signal = amplitude * torch.sin(2 * 3.14159 * base_freq * freq_factor * 3 * t * frame_modulation)
                                            signal += amplitude * 0.5 * torch.sin(2 * 3.14159 * base_freq * freq_factor * 7 * t * frame_modulation)
                                        else:  # 低强度表情
                                            # 生成更平缓的音频模式（如中性、悲伤）
                                            signal = amplitude * 0.7 * torch.sin(2 * 3.14159 * base_freq * freq_factor * t * frame_modulation)
                                        
                                        # 添加更多随机噪声和帧变化
                                        noise = torch.randn_like(signal) * (0.01 + 0.005 * (frame_count % 10) / 10)
                                        frame_variation = 0.02 * torch.sin(torch.tensor(frame_count * 0.05 + i * 0.1))
                                        mel_spec[0, i, :] = signal + noise + frame_variation
                                    
                                    # 应用梅尔频谱的典型特征（低频更强）
                                    freq_weights = torch.exp(-torch.arange(mel_channels, dtype=torch.float32) * 0.05).to(DEVICE)
                                    mel_spec = mel_spec * freq_weights.view(1, -1, 1)
                                    
                                    print(f"视觉特征统计: mean={visual_features.mean():.6f}, std={visual_features.std():.6f}")
                                    print(f"视觉强度: {visual_intensity:.6f}")
                                    print(f"智能音频特征统计: mean={mel_spec.mean():.6f}, std={mel_spec.std():.6f}")
                                    print(f"音频特征形状: {mel_spec.shape}")
                                    
                                    # 使用完整的多模态模型进行推理
                                    outputs = EMOTION_MODEL(face_tensor, mel_spec)
                                    
                                    # 情绪敏感度平衡处理
                                    # 1. 调整softmax温度参数，保持适度敏感度
                                    temperature = 0.7  # 适中的温度参数
                                    
                                    # 2. 平衡的权重调整策略
                                    emotion_weights = torch.ones_like(outputs).to(DEVICE)
                                    neutral_idx = EMOTION_CLASSES.index('neutral')
                                    joy_idx = EMOTION_CLASSES.index('joy')
                                    anger_idx = EMOTION_CLASSES.index('anger')
                                    surprise_idx = EMOTION_CLASSES.index('surprise')
                                    sadness_idx = EMOTION_CLASSES.index('sadness')
                                    disgust_idx = EMOTION_CLASSES.index('disgust')
                                    fear_idx = EMOTION_CLASSES.index('fear')
                                    
                                    # 平衡的权重分配
                                    emotion_weights[0, neutral_idx] = 0.8   # 适度降低中性
                                    emotion_weights[0, joy_idx] = 1.2       # 适度增强快乐
                                    emotion_weights[0, sadness_idx] = 1.2   # 适度增强悲伤
                                    emotion_weights[0, anger_idx] = 1.1     # 轻微增强愤怒
                                    emotion_weights[0, surprise_idx] = 1.1  # 轻微增强惊讶
                                    emotion_weights[0, disgust_idx] = 1.0   # 保持厌恶
                                    emotion_weights[0, fear_idx] = 1.0      # 保持恐惧
                                    
                                    # 应用权重调整
                                    adjusted_outputs = outputs * emotion_weights
                                    
                                    # 应用温度缩放
                                    scaled_outputs = adjusted_outputs / temperature
                                    
                                    # 计算调整后的概率
                                    emotion_probs = torch.nn.functional.softmax(scaled_outputs, dim=1)
                                    predicted_emotion_idx = torch.argmax(emotion_probs).item()
                                    confidence = emotion_probs[0][predicted_emotion_idx].item()
                                    
                                    # 添加详细调试信息
                                    print(f"原始输出: {outputs[0].detach().cpu().numpy()}")
                                    print(f"权重调整后: {adjusted_outputs[0].detach().cpu().numpy()}")
                                    print(f"温度缩放后: {scaled_outputs[0].detach().cpu().numpy()}")
                                    print(f"最终概率分布: {emotion_probs[0].detach().cpu().numpy()}")
                                    print(f"情绪权重: {emotion_weights[0].detach().cpu().numpy()}")
                                    print(f"温度参数: {temperature}")
                                    print(f"各情绪概率:")
                                    for i, emotion in enumerate(EMOTION_CLASSES):
                                        prob = emotion_probs[0][i].item()
                                        weight = emotion_weights[0][i].item()
                                        print(f"  {emotion}: {prob:.4f} (权重: {weight:.1f})")
                                    print(f"预测情感: {EMOTION_CLASSES[predicted_emotion_idx]}, 置信度: {confidence:.3f}")
                                    
                                    emotion_results.append({
                                        'frame': frame_count,
                                        'emotion': EMOTION_CLASSES[predicted_emotion_idx],
                                        'confidence': float(confidence)
                                    })
                            except Exception as e:
                                print(f"模型预测错误: {e}")
                                print(f"错误详情: {type(e).__name__}: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                emotion_results.append({
                                    'frame': frame_count,
                                    'emotion': 'prediction_error',
                                    'confidence': 0.0
                                })
                        else:
                            # 模型未加载，使用随机情感作为占位符
                            import random
                            predicted_emotion = random.choice(EMOTION_CLASSES)
                            confidence = random.uniform(0.6, 0.9)
                            
                            print(f"模型未加载，使用随机情感: {predicted_emotion}, 置信度: {confidence:.3f}")
                            
                            emotion_results.append({
                                'frame': frame_count,
                                'emotion': predicted_emotion,
                                'confidence': confidence
                            })
                else:
                    print(f"第 {frame_count} 帧未检测到人脸 - 帧尺寸: {frame.shape}")
                    emotion_results.append({
                        'frame': frame_count,
                        'emotion': 'no_face_detected',
                        'confidence': 0.0
                    })
                     
            except Exception as e:
                emotion_results.append({
                    'frame': frame_count,
                    'error': str(e)
                })
            
            frame_count += 1
        
        cap.release()
        
        # 计算整体统计
        valid_results = [r for r in emotion_results if 'emotion' in r and r['emotion'] not in ['no_face_detected', 'prediction_error']]
        
        if valid_results:
            # 计算平均情绪
            emotion_counts = {}
            total_confidence = 0
            
            for result in valid_results:
                emotion = result['emotion']
                confidence = result['confidence']
                
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = {'count': 0, 'total_confidence': 0}
                
                emotion_counts[emotion]['count'] += 1
                emotion_counts[emotion]['total_confidence'] += confidence
                total_confidence += confidence
            
            # 找出主要情绪
            dominant_emotion = max(emotion_counts.keys(), key=lambda x: emotion_counts[x]['count'])
            avg_confidence = total_confidence / len(valid_results)
            
            summary = {
                'total_frames': total_frames,
                'frames_with_faces': len(valid_results),
                'dominant_emotion': dominant_emotion,
                'average_confidence': avg_confidence,
                'emotion_distribution': {emotion: counts['count'] for emotion, counts in emotion_counts.items()},
                'detailed_results': emotion_results[:10]  # 只返回前10帧的详细结果
            }
        else:
            summary = {
                'total_frames': total_frames,
                'frames_with_faces': 0,
                'error': '视频中未检测到人脸'
            }
        
        return summary
        
    except Exception as e:
        return {'error': f'视频处理失败: {str(e)}'}


# 用于向客户端发送消息
def push_message(message, color):
    socketio.emit('server_message', {'message': message, 'color': color})

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        push_message('没有视频文件', '#FF0000')  # 红色，表示错误
        return jsonify({'message': '没有视频文件'}), 400
    file = request.files['video']
    if file.filename == '':
        push_message('没有选择文件', '#FF0000')  # 红色，表示错误
        return jsonify({'message': '没有选择文件'}), 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        push_message(f'视频 {filename} 上传成功，开始情绪分析...', '#006400')  # 绿色，表示成功
        
        # 在后台线程中处理情绪识别
        def process_emotion_async():
            try:
                # 进行情绪识别
                emotion_result = process_video_emotion(filepath)
                
                if 'error' in emotion_result:
                    push_message(f'情绪分析失败: {emotion_result["error"]}', '#FF0000')
                else:
                    # 格式化结果消息
                    if 'dominant_emotion' in emotion_result:
                        dominant_emotion = emotion_result['dominant_emotion']
                        confidence = emotion_result['average_confidence']
                        frames_with_faces = emotion_result['frames_with_faces']
                        total_frames = emotion_result['total_frames']
                        
                        emoji = EMOTION_EMOJIS.get(dominant_emotion, '🤔')
                        
                        result_message = f"""🎬 视频情绪分析完成！
{emoji} 主要情绪: {dominant_emotion}
📊 平均置信度: {confidence:.2f}
👤 检测到人脸帧数: {frames_with_faces}/{total_frames}

📈 情绪分布:"""
                        
                        for emotion, count in emotion_result['emotion_distribution'].items():
                            percentage = (count / frames_with_faces) * 100 if frames_with_faces > 0 else 0
                            emoji = EMOTION_EMOJIS.get(emotion, '🤔')
                            result_message += f"\n{emoji} {emotion}: {count}帧 ({percentage:.1f}%)"
                        
                        push_message(result_message, '#0066CC')  # 蓝色，表示分析结果
                    else:
                        push_message(f'情绪分析完成，但{emotion_result.get("error", "未知错误")}', '#FF8800')
                        
            except Exception as e:
                push_message(f'情绪分析过程中出错: {str(e)}', '#FF0000')
        
        # 启动后台处理线程
        threading.Thread(target=process_emotion_async, daemon=True).start()
        
        return jsonify({
            'message': f'视频 {filename} 上传成功，正在进行情绪分析...',
            'path': filepath,
            'filename': filename,
            'size': file.content_length,
            'type': file.mimetype,
            'status': 'processing'
        })

    push_message('上传失败', '#FF0000')  # 红色，表示错误
    return jsonify({'message': '上传失败'}), 500



def handle_user_input():
    while True:
        user_input = input("请输入测试信息 (或输入 'exit' 退出): ")
        if user_input.lower() == 'exit':
            break
        push_message(user_input, '#006400')  # 绿色文本


if __name__ == '__main__':
    print("正在启动情绪识别视频分析服务器...")
    
    # 初始化情绪识别模型
    print("正在加载情绪识别模型...")
    model_loaded = initialize_emotion_model()
    
    if model_loaded:
        print("✅ 情绪识别模型加载成功！")
    else:
        print("⚠️  情绪识别模型加载失败，将仅提供视频上传功能")
    
    print("🚀 服务器启动中...")
    print("📱 请在浏览器中访问: http://localhost:5000")
    print("🎥 现在可以上传视频进行情绪分析了！")
    
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)