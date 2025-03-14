# models/lstm_model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_channels):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_channels, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, input_channels, window_size] -> [batch, window_size, input_channels]
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.decoder(out)
        return out
