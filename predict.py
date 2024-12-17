import tensorflow as tf
import pickle
from utils.preprocess import preprocess_image
from config import Config

def predict(image_path):
    # 加载模型
    model = tf.keras.models.load_model(Config.MODEL_SAVE_PATH)
    
    # 加载字符映射
    with open(Config.CHAR_MAPPINGS_PATH, 'rb') as f:
        char_to_label, label_to_char = pickle.load(f)
    
    # 预处理图片
    img = preprocess_image(image_path, Config.IMAGE_SIZE)
    
    # 预测
    prediction = model.predict(tf.expand_dims(img, axis=0))
    
    # 获取预测结果
    predicted_class = tf.argmax(prediction[0]).numpy()
    
    # 转换为汉字
    predicted_char = label_to_char[predicted_class]
    
    return predicted_char