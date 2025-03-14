import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvLayer(nn.Module):
    """时间卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            (1, kernel_size),
            padding=(0, kernel_size//2)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x shape: (batch_size, num_nodes, in_channels, time_steps)
        x = x.permute(0, 2, 1, 3)  # -> (batch_size, in_channels, num_nodes, time_steps)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3)  # 恢复原始维度顺序
        return x

class GraphConvLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):
        """
        参数:
            x: (batch_size, num_nodes, in_channels, time_steps)
            adj: (batch_size, num_nodes, num_nodes)
        返回:
            output: (batch_size, num_nodes, out_channels, time_steps)
        """
        batch_size, num_nodes, in_channels, time_steps = x.size()
        
        # 对每个时间步进行图卷积
        outputs = []
        for t in range(time_steps):
            # 获取当前时间步的特征
            x_t = x[:, :, :, t]  # (batch_size, num_nodes, in_channels)
            
            # 图卷积操作
            support = torch.matmul(x_t, self.weights)  # (batch_size, num_nodes, out_channels)
            output = torch.matmul(adj, support) + self.bias  # (batch_size, num_nodes, out_channels)
            outputs.append(output.unsqueeze(-1))  # 添加时间维度
        
        # 沿时间维度拼接
        return torch.cat(outputs, dim=-1)  # (batch_size, num_nodes, out_channels, time_steps)

class STGCNBlock(nn.Module):
    """STGCN块：时间卷积-图卷积-时间卷积"""
    def __init__(self, in_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TemporalConvLayer(in_channels, out_channels)
        self.graph_conv = GraphConvLayer(out_channels, out_channels)
        self.temporal2 = TemporalConvLayer(out_channels, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.bn = nn.BatchNorm2d(num_nodes)
        
    def forward(self, x, adj):
        # x shape: (batch_size, num_nodes, in_channels, time_steps)
        residual = x if self.residual is None else self.residual(x.permute(0,2,1,3)).permute(0,2,1,3)
        
        x = self.temporal1(x)
        x = self.graph_conv(x, adj)
        x = self.temporal2(x)
        
        x = x + residual
        x = self.bn(x)
        x = F.relu(x)
        
        return x 