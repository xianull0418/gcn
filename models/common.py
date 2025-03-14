# common.py
import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset

###############################################
# VMD Decomposer
###############################################
class VMDDecomposer:
    # VMD分解器，用于将信号分解为多个模态
    def __init__(self, alpha=2000, tau=0, K=8, DC=0, init=1, tol=1e-7):
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol

    # 对信号进行VMD分解
    def decompose(self, signal):
        T = len(signal)
        f_hat = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftfreq(T, d=1.0)
        freqs = np.fft.fftshift(freqs)
        if self.init == 1:
            omega_plus = np.linspace(0, 0.5, self.K)
        elif self.init == 2:
            omega_plus = np.sort(np.abs(np.random.randn(self.K)))
        else:
            omega_plus = np.zeros(self.K)
        u_hat = np.zeros((self.K, T), dtype=complex)
        lambda_hat = np.zeros(T, dtype=complex)
        uDiff = self.tol + 1
        n = 0
        N_iter = 500
        while uDiff > self.tol and n < N_iter:
            u_hat_prev = u_hat.copy()
            for k in range(self.K):
                sum_others = np.sum(u_hat[np.arange(self.K) != k, :], axis=0)
                denominator = 1 + 2 * self.alpha * (freqs - omega_plus[k])**2
                u_hat[k, :] = (f_hat - sum_others + lambda_hat / 2) / denominator
                if not (self.DC and k == 0):
                    numerator = np.sum(freqs * (np.abs(u_hat[k, :])**2))
                    denominator_omega = np.sum(np.abs(u_hat[k, :])**2) + 1e-10
                    omega_plus[k] = numerator / denominator_omega
            lambda_hat = lambda_hat + self.tau * (np.sum(u_hat, axis=0) - f_hat)
            uDiff = np.linalg.norm(u_hat - u_hat_prev) / (np.linalg.norm(u_hat_prev) + 1e-10)
            n += 1
        u_hat = np.fft.ifftshift(u_hat, axes=1)
        modes = np.real(np.fft.ifft(u_hat, axis=1))
        return modes

###############################################
# Positional Encoding
###############################################
class PositionalEncoding(nn.Module):
    # 位置编码，用于将位置信息编码到模型的输入中
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    # 前向传播，将位置编码添加到输入张量
    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]

###############################################
# DataProcessor
###############################################
class DataProcessor:
    # 数据处理器，负责加载、合并、归一化和VMD处理水和雨的数据
    def __init__(self, water_path, rain_path, date_col='TM',
                 water_val_col='Z', rain_val_col='DRP',
                 window_size=72, K_wl=9, K_rf=8):
        self.water_path = water_path
        self.rain_path = rain_path
        self.date_col = date_col
        self.water_val_col = water_val_col
        self.rain_val_col = rain_val_col
        self.window_size = window_size
        self.K_wl = K_wl
        self.K_rf = K_rf
        self.scaler_wl = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_rf = MinMaxScaler(feature_range=(-1, 1))
        self.vmd_wl = VMDDecomposer(K=self.K_wl)
        self.vmd_rf = VMDDecomposer(K=self.K_rf)

    # 加载并合并水和雨的数据
    def load_and_merge_data(self):
        water_df = pd.read_csv(self.water_path)
        rain_df = pd.read_csv(self.rain_path)
        water_df.rename(columns={self.date_col: 'date'}, inplace=True)
        rain_df.rename(columns={self.date_col: 'date'}, inplace=True)
        water_df['date'] = pd.to_datetime(water_df['date'])
        rain_df['date'] = pd.to_datetime(rain_df['date'])
        water_df.sort_values("date", inplace=True)
        rain_df.sort_values("date", inplace=True)
        merged_data = pd.merge(water_df, rain_df, on="date", how="inner",
                               suffixes=('_wl', '_rf'))
        merged_data.ffill(inplace=True)
        return merged_data

    # 对数据进行归一化处理
    def scale_data(self, df):
        df['value_wl'] = self.scaler_wl.fit_transform(df[[self.water_val_col]])
        df['value_rf'] = self.scaler_rf.fit_transform(df[[self.rain_val_col]])
        return df

    # 应用VMD分解水和雨的信号
    def apply_vmd(self, df):
        wl_array = df['value_wl'].values
        rf_array = df['value_rf'].values
        wl_imfs = self.vmd_wl.decompose(wl_array)
        rf_imfs = self.vmd_rf.decompose(rf_array)
        return wl_imfs, rf_imfs

    # 创建滑动窗口
    def create_sliding_windows(self, wl_imfs, rf_imfs):
        X_water, X_rain, Y = [], [], []
        total_length = wl_imfs.shape[1]
        for i in range(total_length - self.window_size):
            wl_window = wl_imfs[:, i : i + self.window_size]
            rf_window = rf_imfs[:, i : i + self.window_size]
            target = wl_imfs[0, i + self.window_size]
            X_water.append(wl_window)
            X_rain.append(rf_window)
            Y.append(target)
        X_water = torch.tensor(np.array(X_water), dtype=torch.float32)
        X_rain = torch.tensor(np.array(X_rain), dtype=torch.float32)
        Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)
        return X_water, X_rain, Y

    # 处理数据的主函数
    def process(self):
        df = self.load_and_merge_data()
        df = self.scale_data(df)
        wl_imfs, rf_imfs = self.apply_vmd(df)
        return self.create_sliding_windows(wl_imfs, rf_imfs)

###############################################
# Merged Dataset for models using merged channels
###############################################
class MergedWaterRainDataset(Dataset):
    # 合并水和雨数据集，用于模型输入
    def __init__(self, merged, targets):
        self.merged = merged
        self.targets = targets

    # 返回数据集的大小
    def __len__(self):
        return len(self.targets)

    # 获取指定索引的数据
    def __getitem__(self, idx):
        return self.merged[idx], self.targets[idx]

###############################################
# Evaluation Metrics
###############################################
def compute_rmse(y_true, y_pred):
    # 计算均方根误差（RMSE）
    return np.sqrt(np.mean((y_true - y_pred)**2))

def compute_nse(y_true, y_pred):
    # 计算纳什效率系数（NSE）
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - numerator / denominator if denominator != 0 else float('nan')

def compute_mae(y_true, y_pred):
    # 计算平均绝对误差（MAE）
    return np.mean(np.abs(y_true - y_pred))
