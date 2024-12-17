from flask import Flask, render_template, request, jsonify, Response
import os
from predict import predict
from train import train_model
from werkzeug.utils import secure_filename
import threading
import queue
import time
from flask_sock import Sock
import json
import tensorflow as tf

app = Flask(__name__)
sock = Sock(app)

# 配置
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'static/datasets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip', 'gnt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# 全局变量存储训练状态
training_progress = {
    'status': 'idle',  # idle, training, completed, error
    'progress': 0,
    'message': '',
    'current_epoch': 0,
    'total_epochs': 0,
    'metrics': {
        'current_epoch': 0,
        'total_epochs': 0,
        'loss': None,
        'accuracy': None,
        'val_loss': None,
        'val_accuracy': None
    },
    'should_pause': False,  # 添加暂停状态
    'details': ''
}

# 添加训练控制状态
training_control = {
    'should_pause': False,
    'should_stop': False
}


def allowed_file(filename, types=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in types


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train_page():
    return render_template('train.html')


@app.route('/recognize', methods=['POST'])
def recognize_image():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg'}):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            training_progress['status'] = 'predicting'
            result = predict(filepath)
            training_progress['status'] = 'idle'
            return jsonify({
                'success': True,
                'character': result,
                'image_url': f'/static/uploads/{filename}'
            })
        except Exception as e:
            training_progress['status'] = 'error'
            return jsonify({'error': f'识别出错: {str(e)}'}), 500

    return jsonify({'error': '不支持的文件类型'}), 400


def train_process(dataset_path, epochs=50, batch_size=32, use_gpu=False):
    try:
        training_progress['status'] = 'training'
        training_progress['message'] = '正在准备训练数据...'
        training_progress['progress'] = 0
        training_progress['metrics'] = {
            'current_epoch': 0,
            'total_epochs': epochs,
            'loss': None,
            'accuracy': None,
            'val_loss': None,
            'val_accuracy': None
        }

        def progress_callback(epoch, total_epochs, batch, total_batches, loss, metrics=None, details=None):
            training_progress['current_epoch'] = epoch
            training_progress['total_epochs'] = total_epochs
            training_progress['progress'] = (epoch * 100) // total_epochs
            training_progress['message'] = f'训练中 - Epoch {epoch}/{total_epochs}'
            if metrics:
                training_progress['metrics'] = metrics
            if details:
                training_progress['details'] = details

        train_model(
            dataset_path,
            progress_callback,
            epochs=epochs,
            batch_size=batch_size,
            use_gpu=use_gpu,
            training_control=training_control  # 传递训练控制字典
        )

        training_progress['status'] = 'completed'
        training_progress['progress'] = 100
        training_progress['message'] = '训练完成！'

    except Exception as e:
        training_progress['status'] = 'error'
        training_progress['message'] = f'训练出错: {str(e)}'


@sock.route('/ws/train')
def ws_train(ws):
    """WebSocket处理训练进度更新"""
    try:
        while True:
            # 发送当前训练状态
            ws.send(json.dumps(training_progress))
            # 如果训练完成或出错,关闭连接
            if training_progress['status'] in ['completed', 'error']:
                break
            time.sleep(1)  # 每秒更新一次
    except Exception as e:
        print(f"WebSocket error: {str(e)}")


@app.route('/start-training', methods=['POST'])
def start_training():
    """启动模型训练"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供训练参数'}), 400

        # 获取并验证参数
        dataset_path = data.get('dataset_path', 'static/datasets')
        epochs = int(data.get('epochs', 50))  # 确保是整数
        batch_size = int(data.get('batch_size', 32))  # 确保是整数

        # 参数验证
        if epochs <= 0:
            return jsonify({'error': '训练轮数必须大于0'}), 400
        if batch_size <= 0:
            return jsonify({'error': '批次大小必须大于0'}), 400

        # 检查数据集路径
        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')

        if not os.path.exists(train_path):
            return jsonify({'error': 'train目录不存在'}), 400
        if not os.path.exists(test_path):
            return jsonify({'error': 'test目录不存在'}), 400

        # 添加 GPU 选项
        use_gpu = data.get('use_gpu', False)

        # 在新线程中启动训练，传入自定义参数
        thread = threading.Thread(
            target=train_process,
            args=(dataset_path,),
            kwargs={
                'epochs': epochs,
                'batch_size': batch_size,
                'use_gpu': use_gpu  # 传递 GPU 选项
            }
        )
        thread.start()

        return jsonify({
            'success': True,
            'message': '训练已开始',
            'params': {
                'dataset_path': dataset_path,
                'epochs': epochs,
                'batch_size': batch_size
            }
        })

    except ValueError as e:
        return jsonify({'error': f'参数错误: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'启动训练失败: {str(e)}'}), 500


@app.route('/training-status')
def get_training_status():
    return jsonify(training_progress)


@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/check-gpu')
def check_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # 获取 GPU 信息
            gpu_info = []
            for gpu in gpus:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_info.append(f"{gpu_details.get('device_name', 'Unknown GPU')}")
            return jsonify({
                'gpu_available': True,
                'gpu_info': ', '.join(gpu_info)
            })
        else:
            return jsonify({
                'gpu_available': False,
                'gpu_info': None
            })
    except Exception as e:
        return jsonify({
            'gpu_available': False,
            'error': str(e)
        })


@app.route('/check-gpu-environment')
def check_gpu_environment():
    try:
        environment_status = {'gpu_available': False, 'driver_version': None, 'cuda_available': False,
                              'cuda_version': None, 'cudnn_available': False, 'cudnn_version': None,
                              'tensorflow_gpu': False, 'tensorflow_version': tf.__version__, 'details': []}

        # 查 TensorFlow 版本
        environment_status['details'].append(f"TensorFlow 版本: {tf.__version__}")

        # 检查 GPU 是否可用
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            environment_status['gpu_available'] = True
            # 获取 GPU 信息
            for gpu in gpus:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                environment_status['details'].append(f"检测到 GPU: {gpu_details.get('device_name', 'Unknown GPU')}")
        else:
            environment_status['details'].append("未检测到可用的 GPU")

        # 检查 CUDA 是否可用
        try:
            cuda_version = tf.sysconfig.get_build_info()['cuda_version']
            environment_status['cuda_available'] = True
            environment_status['cuda_version'] = cuda_version
            environment_status['details'].append(f"CUDA 版本: {cuda_version}")
        except:
            environment_status['details'].append("未检测到 CUDA")

        # 检查 cuDNN 是否可用
        try:
            cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
            environment_status['cudnn_available'] = True
            environment_status['cudnn_version'] = cudnn_version
            environment_status['details'].append(f"cuDNN 版本: {cudnn_version}")
        except:
            environment_status['details'].append("未检测到 cuDNN")

        # 检查是否为 GPU 版本的 TensorFlow
        environment_status['tensorflow_gpu'] = tf.test.is_built_with_cuda()
        if environment_status['tensorflow_gpu']:
            environment_status['details'].append("已安装 GPU 版本的 TensorFlow")
        else:
            environment_status['details'].append("未安装 GPU 版本的 TensorFlow")

        return jsonify(environment_status)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'details': ['检测过程出现错误']
        })


@app.route('/pause-training', methods=['POST'])
def pause_training():
    training_control['should_pause'] = not training_control['should_pause']
    training_progress['should_pause'] = training_control['should_pause']  # 更新训练状态中的暂停状态
    return jsonify({
        'success': True,
        'paused': training_control['should_pause']
    })


@app.route('/stop-training', methods=['POST'])
def stop_training():
    training_control['should_stop'] = True
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True, port=5005)
