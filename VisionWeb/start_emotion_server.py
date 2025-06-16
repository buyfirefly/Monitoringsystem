#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪识别视频分析服务器启动脚本
集成了VisionWeb界面和多模态情绪识别系统

使用方法:
1. 确保已安装所有依赖包
2. 运行此脚本: python start_emotion_server.py
3. 在浏览器中访问: http://localhost:5000
4. 上传视频或录制视频进行情绪分析
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖包是否已安装"""
    required_packages = [
        'flask',
        'flask-socketio',
        'flask-cors',
        'torch',
        'cv2',
        'numpy',
        'face_alignment'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'face_alignment':
                import face_alignment
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请先安装缺少的依赖包:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_model_files():
    """检查模型文件是否存在"""
    model_path = Path('../multimodal-emotion-recognition-system-master/emonet/pretrained')
    
    if not model_path.exists():
        print("⚠️  未找到预训练模型文件夹")
        print(f"   期望路径: {model_path.absolute()}")
        print("   请确保EmoNet预训练模型已下载到正确位置")
        return False
    
    # 检查是否有模型文件
    model_files = list(model_path.glob('*.pth'))
    if not model_files:
        print("⚠️  预训练模型文件夹为空")
        print("   请下载EmoNet预训练模型文件")
        return False
    
    print(f"✅ 找到 {len(model_files)} 个模型文件")
    return True

def main():
    print("🚀 情绪识别视频分析服务器启动检查")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查模型文件
    model_available = check_model_files()
    
    if not model_available:
        print("\n⚠️  模型文件检查失败，但服务器仍可启动（仅提供视频上传功能）")
        response = input("是否继续启动服务器？(y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎯 启动服务器...")
    
    # 启动服务器
    try:
        from server import app, socketio, initialize_emotion_model
        
        print("正在加载情绪识别模型...")
        model_loaded = initialize_emotion_model()
        
        if model_loaded:
            print("✅ 情绪识别模型加载成功！")
        else:
            print("⚠️  情绪识别模型加载失败，将仅提供视频上传功能")
        
        print("\n🌐 服务器信息:")
        print("   地址: http://localhost:5000")
        print("   功能: 视频录制 + AI情绪分析")
        print("\n📝 使用说明:")
        print("   1. 在浏览器中打开上述地址")
        print("   2. 点击'开始录制'录制视频，或使用'手动上传'上传视频文件")
        print("   3. 系统将自动分析视频中的情绪并显示结果")
        print("\n按 Ctrl+C 停止服务器")
        print("=" * 50)
        
        socketio.run(app, debug=False, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()