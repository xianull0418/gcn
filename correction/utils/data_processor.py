import pandas as pd
import numpy as np
from pathlib import Path
from correction.config import Config

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.data_path = Path(config.DATA_PATH)
        
    def load_data(self):
        """加载所有原始数据"""
        # 加载水位数据
        water_level_df = pd.read_csv(
            self.data_path / self.config.WATER_LEVEL_FILE,
            parse_dates=['TM']
        )
        
        # 加载降雨数据
        rainfall_df = pd.read_csv(
            self.data_path / self.config.RAINFALL_FILE,
            parse_dates=['TM']
        )
        
        # 加载模型预测结果
        result_df = pd.read_csv(
            self.data_path / self.config.RESULT_FILE,
            parse_dates=['TM']
        )
        
        return water_level_df, rainfall_df, result_df
    
    def create_water_level_bins(self, water_level_data):
        """创建水位分层"""
        min_level = water_level_data['Z'].min()
        max_level = water_level_data['Z'].max()
        bins = np.linspace(min_level, max_level, self.config.WATER_LEVEL_BINS + 1)
        return bins
    
    def process_data(self):
        """处理数据并准备模型输入"""
        water_level_df, rainfall_df, result_df = self.load_data()
        
        # 合并数据
        merged_df = pd.merge(result_df, rainfall_df[['TM', 'DRP']], 
                           on='TM', how='left')
        
        # 创建水位分层
        water_bins = self.create_water_level_bins(water_level_df)
        
        # 准备时序特征
        features = []
        labels = []
        
        # 确保所有数据都是float32类型
        model_columns = ['DoubleEncoderTransformer', 'Transformer', 'LSTM', 'GRU', 'CNN']
        merged_df['Actual'] = merged_df['Actual'].astype(np.float32)
        merged_df['DRP'] = merged_df['DRP'].astype(np.float32)
        for col in model_columns:
            merged_df[col] = merged_df[col].astype(np.float32)
        
        for i in range(len(merged_df) - self.config.TIME_WINDOW):
            # 获取时间窗口内的数据
            window_data = merged_df.iloc[i:i + self.config.TIME_WINDOW]
            target = merged_df.iloc[i + self.config.TIME_WINDOW]
            
            # 计算预测误差
            model_errors = (target[model_columns].values - target['Actual']).astype(np.float32)
            
            # 确保特征数据是float32类型
            feature_data = window_data[['Actual', 'DRP']].values.astype(np.float32)
            
            features.append(feature_data)
            labels.append(model_errors)
        
        # 转换为numpy数组并确保类型正确
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        return features, labels, water_bins 