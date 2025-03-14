import numpy as np
import torch

def rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """平均绝对百分比误差"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Metrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """计算多个评估指标"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
            
        metrics = {
            'rmse': rmse(y_true, y_pred),
            'mae': mae(y_true, y_pred),
            'mape': mape(y_true, y_pred)
        }
        return metrics
    
    @staticmethod
    def print_metrics(metrics):
        """打印评估指标"""
        print("评估结果:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MAPE: {metrics['mape']:.4f}%") 