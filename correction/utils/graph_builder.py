import numpy as np
import torch
from correction.config import Config

class GraphBuilder:
    def __init__(self, config: Config):
        self.config = config
        
    def build_water_level_graph(self, water_bins):
        """构建水位层次图"""
        num_nodes = self.config.WATER_LEVEL_BINS
        # 初始化邻接矩阵
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # 构建相邻水位层之间的连接
        for i in range(num_nodes-1):
            adj_matrix[i, i+1] = 1
            adj_matrix[i+1, i] = 1
            
        # 自连接
        adj_matrix = adj_matrix + np.eye(num_nodes)
        
        # 计算度矩阵
        degree_matrix = np.sum(adj_matrix, axis=1)
        degree_matrix = np.diag(np.power(degree_matrix, -0.5))
        
        # 归一化邻接矩阵
        norm_adj = degree_matrix @ adj_matrix @ degree_matrix
        
        return torch.FloatTensor(norm_adj)
    
    def get_water_level_index(self, water_level, water_bins):
        """获取水位值对应的层次索引"""
        return np.digitize(water_level, water_bins) - 1
    
    def build_dynamic_graph(self, features, water_bins):
        """构建动态图特征"""
        batch_size, seq_len, _ = features.shape
        num_nodes = self.config.WATER_LEVEL_BINS
        
        # 先创建numpy数组
        graph_features_np = np.zeros((batch_size, seq_len, num_nodes, 2), dtype=np.float32)
        
        for b in range(batch_size):
            for t in range(seq_len):
                water_level = features[b, t, 0]  # 水位值
                rainfall = features[b, t, 1]     # 降雨量
                
                # 获取水位对应的层次索引
                level_idx = self.get_water_level_index(water_level, water_bins)
                
                # 将特征分配到对应的节点
                graph_features_np[b, t, level_idx, 0] = water_level
                graph_features_np[b, t, level_idx, 1] = rainfall
        
        # 最后一次性转换为PyTorch张量
        return torch.FloatTensor(graph_features_np) 