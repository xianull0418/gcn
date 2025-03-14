import torch
import torch.nn as nn
from .layers import STGCNBlock

class STGCN(nn.Module):
    def __init__(self, config):
        super(STGCN, self).__init__()
        self.config = config
        
        # 模型参数
        num_nodes = config.WATER_LEVEL_BINS
        input_dim = 2  # 水位和降雨量
        hidden_dim = config.HIDDEN_CHANNELS
        output_dim = 5  # 5个模型的误差校正
        
        # 确保hidden_dim能被num_heads整除
        num_heads = 4
        self.hidden_dim = (hidden_dim // num_heads) * num_heads  # 调整hidden_dim为能被num_heads整除的数
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # STGCN层
        self.st_blocks = nn.ModuleList([
            STGCNBlock(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_nodes=num_nodes
            ) for _ in range(config.NUM_LAYERS)
        ])
        
        # 注意力前的投影层
        self.attention_proj = nn.Linear(self.hidden_dim * num_nodes, self.hidden_dim)
        
        # 时间注意力层
        self.time_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads
        )
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
    def forward(self, x, adj):
        """
        参数:
            x: 输入特征 (batch_size, seq_len, num_nodes, input_dim)
            adj: 邻接矩阵 (num_nodes, num_nodes)
        返回:
            output: 预测的误差校正值 (batch_size, output_dim)
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        
        # 输入投影
        x = self.input_proj(x)  # -> (batch_size, seq_len, num_nodes, hidden_dim)
        
        # 调整维度顺序以适应STGCN块
        x = x.permute(0, 2, 3, 1)  # -> (batch_size, num_nodes, hidden_dim, seq_len)
        
        # 通过STGCN块
        for st_block in self.st_blocks:
            x = st_block(x, adj)
            
        # 调整维度用于时间注意力
        x = x.permute(0, 3, 1, 2)  # -> (batch_size, seq_len, num_nodes, hidden_dim)
        x = x.reshape(batch_size, seq_len, -1)  # -> (batch_size, seq_len, num_nodes*hidden_dim)
        
        # 投影到正确的维度
        x = self.attention_proj(x)  # -> (batch_size, seq_len, hidden_dim)
        
        # 调整维度以适应MultiheadAttention
        x = x.transpose(0, 1)  # -> (seq_len, batch_size, hidden_dim)
        
        # 应用时间注意力
        x_attend, _ = self.time_attention(x, x, x)
        
        # 调整回原来的维度顺序
        x = x_attend.transpose(0, 1)  # -> (batch_size, seq_len, hidden_dim)
        
        # 获取最后一个时间步的特征
        x = x[:, -1, :]  # -> (batch_size, hidden_dim)
        
        # 输出层
        output = self.output_proj(x)  # -> (batch_size, output_dim)
        
        return output 