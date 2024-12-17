class Config:
    # 数据集配置
    DATASET_PATH = "data"
    IMAGE_SIZE = (64, 64)  # 统一图片大小
    BATCH_SIZE = 32
    
    # 模型配置
    NUM_CLASSES = 3755  # 常用汉字数量
    LEARNING_RATE = 0.001
    EPOCHS = 50

    # 训练配置
    DEFAULT_PATH = "models"
    DEFAULT_PATH = "models/default"
    MODEL_SAVE_PATH = DEFAULT_PATH + "/chinese_ocr_model.h5"
    CHAR_MAPPINGS_PATH = DEFAULT_PATH + '/char_mappings.pkl'
    
    # 预处理配置
    CHANNELS = 1  # 灰度图 
    
    # 百度OCR API配置
    BAIDU_OCR_API_KEY = "你的API Key"
    BAIDU_OCR_SECRET_KEY = "你的Secret Key"