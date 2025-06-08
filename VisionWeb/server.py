from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import threading
import time
import shutil

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
        push_message(f'视频切片 {filename} 上传成功!', '#006400')  # 绿色，表示成功
        return jsonify({
            'message': f'视频切片 {filename} 上传成功!',
            'path': filepath,
            'filename': filename,
            'size': file.content_length,
            'type': file.mimetype
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
    # print("【注意】服务器每次启动，都需要回车一次！否则客户端会一直加载等待！")
    # # 启动服务器前，启动一个线程处理用户输入（测试使用，实际中用不到，测试时需要二次发送才有效）
    # input_thread = threading.Thread(target=handle_user_input)
    # input_thread.daemon = True  # 设置为守护线程，确保主线程退出时子线程也退出
    # input_thread.start()
    # # 添加短暂的延迟，确保线程启动完成
    # time.sleep(0.2)
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)