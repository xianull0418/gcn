import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd

class BaselineModels:
    def __init__(self, config):
        self.config = config
        
    def prepare_data(self, features, labels):
        """准备用于传统模型的数据"""
        # 将特征展平为2D数组
        n_samples = features.shape[0]
        n_features = np.prod(features.shape[1:])
        X = features.reshape(n_samples, n_features)
        y = labels
        return X, y
    
    class ARModel:
        """自回归模型"""
        def __init__(self, lags=5):
            self.lags = lags
            self.models = []  # 每个输出维度一个AR模型
            
        def fit(self, X, y):
            # 对每个输出维度训练一个AR模型
            for i in range(y.shape[1]):
                model = AutoReg(y[:, i], lags=self.lags)
                self.models.append(model.fit())
            
        def predict(self, X):
            predictions = []
            for model in self.models:
                pred = model.predict(start=len(model.data.orig_endog),
                                  end=len(model.data.orig_endog))
                predictions.append(pred)
            return np.column_stack(predictions)
    
    class KNNModel:
        """K近邻回归模型"""
        def __init__(self, n_neighbors=5):
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
            
        def fit(self, X, y):
            self.model.fit(X, y)
            
        def predict(self, X):
            return self.model.predict(X)
    
    class BPModel:
        """BP神经网络模型"""
        def __init__(self, hidden_layer_sizes=(100, 50)):
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=1000,
                random_state=42
            )
            
        def fit(self, X, y):
            self.model.fit(X, y)
            
        def predict(self, X):
            return self.model.predict(X)
    
    def train_and_evaluate(self, train_features, train_labels, val_features, val_labels):
        """训练和评估所有基准模型"""
        # 准备数据
        X_train, y_train = self.prepare_data(train_features, train_labels)
        X_val, y_val = self.prepare_data(val_features, val_labels)
        
        # 初始化模型
        models = {
            'AR': self.ARModel(lags=24),  # 使用24小时的滞后
            'KNN': self.KNNModel(n_neighbors=5),
            'BP': self.BPModel(hidden_layer_sizes=(100, 50))
        }
        
        results = {}
        for name, model in models.items():
            print(f"\n训练 {name} 模型...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测和评估
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # 计算评估指标
            from correction.utils.metrics import Metrics
            train_metrics = Metrics.calculate_metrics(y_train, train_pred)
            val_metrics = Metrics.calculate_metrics(y_val, val_pred)
            
            results[name] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model': model
            }
            
            print(f"\n{name} 模型评估结果:")
            print("训练集:")
            Metrics.print_metrics(train_metrics)
            print("\n验证集:")
            Metrics.print_metrics(val_metrics)
            
        return results 