# models/cnn_model.py
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_channels, window_size, num_filters=64, kernel_size=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=num_filters,
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: [batch, input_channels, window_size]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        out = self.decoder(x)
        return out
