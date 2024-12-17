import tensorflow as tf
from models.cnn_model import build_model
from utils.dataset import load_dataset
from config import Config
import os
import logging
import pickle
import time


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def train_model(dataset_path, progress_callback=None, epochs=50, batch_size=32, use_gpu=False, training_control=None):
    """
    训练模型的函数
    
    Args:
        dataset_path: 数据集根目录路径
        progress_callback: 回调函数，用于更新训练进度
        epochs: 训练轮数
        batch_size: 批次大小
        use_gpu: 是否使用 GPU
        training_control: 训练控制字典，用于暂停和停止训练
    """
    setup_logging()
    logging.info(f"开始训练过程 - {'GPU' if use_gpu else 'CPU'} 模式")

    # 如果没有传入训练控制字典，创建一个默认的
    if training_control is None:
        training_control = {'should_pause': False, 'should_stop': False}

    # GPU 配置
    try:
        if not use_gpu:
            logging.info("使用 CPU 训练")
            tf.config.set_visible_devices([], 'GPU')
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # 启用内存增长
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"使用 GPU 训练: {len(gpus)} 个 GPU 可用")
                    
                    # 打印 GPU 详细信息
                    for gpu in gpus:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        logging.info(f"GPU 详情: {gpu_details}")
                        
                except RuntimeError as e:
                    logging.error(f"GPU 配置错误: {e}")
                    tf.config.set_visible_devices([], 'GPU')
                    logging.info("回退到 CPU 训练")
            else:
                logging.warning("未检测到 GPU，使用 CPU 训练")
                tf.config.set_visible_devices([], 'GPU')
                
        # 验证当前设备
        devices = tf.config.list_physical_devices()
        logging.info(f"可用设备: {devices}")
        
    except Exception as e:
        logging.error(f"设备配置错误: {e}")
        raise

    # 创建日志目录
    os.makedirs('logs/training', exist_ok=True)

    try:
        # 加载训练数据
        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')
        
        logging.info(f"正在从 {train_path} 加载训练数据集")
        train_data, train_labels, char_to_label, label_to_char = load_dataset(train_path)

        if len(train_data) == 0:
            raise ValueError("没有成功加载任何训练数据")

        logging.info(f"成功加载训练数据: {len(train_data)} 个样本")

        # 加载测试数据
        logging.info(f"正在从 {test_path} 加载测试数据集")
        test_data, test_labels, _, _ = load_dataset(test_path, char_to_label)

        if len(test_data) == 0:
            raise ValueError("没有成功加载任何测试数据")

        logging.info(f"成功加载测试数据: {len(test_data)} 个样本")

        # 保存字符映射表
        mappings_path = Config.CHAR_MAPPINGS_PATH
        os.makedirs(os.path.dirname(mappings_path), exist_ok=True)
        with open(mappings_path, 'wb') as f:
            pickle.dump((char_to_label, label_to_char), f)
        logging.info("已保存字符映射表")

        # 构建模型
        model = build_model(
            input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], Config.CHANNELS),
            num_classes=Config.NUM_CLASSES
        )
        logging.info("模型构建完成")

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logging.info("模型编译完成")

        # 创建训练回调
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                # 检查是否应该停止训练
                if training_control['should_stop']:
                    self.model.stop_training = True
                    logging.info("训练已手动停止")
                    return

                # 处理暂停
                while training_control['should_pause'] and not training_control['should_stop']:
                    time.sleep(1)  # 暂停时每秒检查一次状态
                    if progress_callback:
                        progress_callback(
                            epoch + 1,
                            epochs,
                            0,
                            0,
                            None,
                            metrics=self.model.history.history if hasattr(self.model, 'history') else None,
                            details="训练已暂停..."
                        )

                monitor_memory()
                
            def on_epoch_end(self, epoch, logs=None):
                monitor_memory()
                if progress_callback:
                    progress_callback(
                        epoch + 1,
                        epochs,
                        0,
                        0,
                        logs.get('loss', 0),
                        metrics={
                            'current_epoch': epoch + 1,
                            'total_epochs': epochs,
                            'loss': logs.get('loss', 0),
                            'accuracy': logs.get('accuracy', 0),
                            'val_loss': logs.get('val_loss', 0),
                            'val_accuracy': logs.get('val_accuracy', 0)
                        },
                        details=f"Epoch {epoch + 1}/{epochs} - "
                               f"loss: {logs.get('loss', 0):.4f} - "
                               f"accuracy: {logs.get('accuracy', 0):.4f} - "
                               f"val_loss: {logs.get('val_loss', 0):.4f} - "
                               f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
                    )
            
            def on_batch_end(self, batch, logs=None):
                if progress_callback and logs:
                    total_batches = len(train_data) // batch_size
                    # 更新当前批次的训练指标
                    progress_callback(
                        self.model.history.epoch[-1] + 1 if self.model.history.epoch else 1,
                        epochs,
                        batch + 1,
                        total_batches,
                        logs.get('loss', 0),
                        metrics={
                            'current_epoch': self.model.history.epoch[-1] + 1 if self.model.history.epoch else 1,
                            'total_epochs': epochs,
                            'loss': logs.get('loss', 0),
                            'accuracy': logs.get('accuracy', 0),
                            'val_loss': self.model.history.history.get('val_loss', [0])[-1] if self.model.history.history.get('val_loss') else 0,
                            'val_accuracy': self.model.history.history.get('val_accuracy', [0])[-1] if self.model.history.history.get('val_accuracy') else 0
                        },
                        details=f"Epoch {self.model.history.epoch[-1] + 1 if self.model.history.epoch else 1}/{epochs} "
                               f"- Batch {batch + 1}/{total_batches} "
                               f"- loss: {logs.get('loss', 0):.4f} "
                               f"- accuracy: {logs.get('accuracy', 0):.4f}"
                    )

        # 训练模型
        history = model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_data, test_labels),
            callbacks=[TrainingCallback()]
        )

        # 保存模型
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
        model.save(Config.MODEL_SAVE_PATH)
        logging.info(f"模型已保存到 {Config.MODEL_SAVE_PATH}")

        return history

    except Exception as e:
        logging.error(f"训练过程中出现错误: {str(e)}")
        raise


def train():
    """命令行训练入口"""
    train_model(Config.DATASET_PATH)


def monitor_memory():
    try:
        if tf.config.list_physical_devices('GPU'):
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            logging.info(f"GPU 内存使用: {memory_info}")
    except:
        pass


if __name__ == "__main__":
    train()
