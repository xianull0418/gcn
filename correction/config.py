from pathlib import Path

class Config:
    # 获取项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent  # 从config.py文件位置向上两级到项目根目录
    
    # 数据相关
    DATA_PATH = str(PROJECT_ROOT / "correction" / "data" / "raw")
    PROCESSED_PATH = str(PROJECT_ROOT / "correction" / "data" / "processed")
    WATER_LEVEL_FILE = "水位_东山.csv"
    RAINFALL_FILE = "雨量_东山.csv"
    RESULT_FILE = "东山_result.csv"
    
    # 模型参数
    WATER_LEVEL_BINS = 10  # 水位分层数
    TIME_WINDOW = 24      # 时间窗口大小（小时）
    HIDDEN_CHANNELS = 64  # 确保是4的倍数
    NUM_LAYERS = 3        # STGCN层数
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = "cuda"  # 或 "cpu"
    
    # PSO-GWO参数
    POPULATION_SIZE = 10     # 减小种群大小
    MAX_ITERATIONS = 30      # 减少迭代次数
    LAMBDA = 0.6         # 调节系数
    ALPHA = 0.5         # 权重因子
    BETA = 0.5          # 权重因子
    
    # 添加评估参数
    EVAL_EPOCHS = 3         # 减少评估时的训练轮数
    
    # 模型保存
    MODEL_SAVE_PATH = str(PROJECT_ROOT / "correction" / "models" / "saved")
    
    # 随机种子
    SEED = 42 