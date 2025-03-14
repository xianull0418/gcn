# models/transformer_model.py
import torch.nn as nn
from models.common import PositionalEncoding

class TransformerModel(nn.Module):
    # TransformerModel类，用于定义基于Transformer的模型
    def __init__(self, d_model, nhead, num_layers, input_channels, window_size):
        super().__init__()
        self.linear_proj = nn.Linear(input_channels, d_model)  
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)  
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  
        self.decoder = nn.Linear(d_model, 1)  

    # 前向传播方法，接受输入并生成预测输出
    def forward(self, x):
        # x: [batch, input_channels, window_size]
        x = x.permute(2, 0, 1)  #
        x = self.linear_proj(x)  
        x = self.pos_encoder(x)  
        x = self.transformer_encoder(x)  
        out = x[-1, :, :]  
        out = self.decoder(out)  
        return out
