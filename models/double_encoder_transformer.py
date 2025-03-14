# models/double_encoder_transformer.py
import torch
import torch.nn as nn
import numpy as np
from models.common import PositionalEncoding

class DoubleEncoderTransformer(nn.Module):
    # DoubleEncoderTransformer模型，包含输入投影、位置编码、Transformer编码器、跨域注意力机制和解码器
    def __init__(self, d_model, nhead, K_wl, K_rf):
        super().__init__()
        self.d_model = d_model
        self.input_proj_water = nn.Linear(K_wl, d_model)
        self.input_proj_rain = nn.Linear(K_rf, d_model)
        self.pos_encoder_water = PositionalEncoding(d_model)
        self.pos_encoder_rain = PositionalEncoding(d_model)
        self.encoder_water = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder_rain = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.decoder = nn.Linear(d_model, 1)

    # 前向传播方法，通过水和雨的输入生成预测结果
    def forward(self, water_input, rain_input):
        water_input = water_input.permute(2, 0, 1)
        rain_input = rain_input.permute(2, 0, 1)
        water_embed = self.input_proj_water(water_input)
        rain_embed = self.input_proj_rain(rain_input)
        water_embed = self.pos_encoder_water(water_embed)
        rain_embed = self.pos_encoder_rain(rain_embed)
        water_encoded = self.encoder_water(water_embed)
        rain_encoded = self.encoder_rain(rain_embed)
        attn_output, _ = self.cross_attn(water_encoded, rain_encoded, rain_encoded)
        final_feature = attn_output[-1, :, :]
        prediction = self.decoder(final_feature)
        return prediction


class DoubleEncoderTrainer:
    # DoubleEncoder模型的训练器，包含训练和评估方法
    def __init__(self, model, train_loader, val_loader, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.device = device

    # 评估方法，计算模型在数据集上的损失
    def evaluate(self, loader):
        self.model.eval()
        losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for water_batch, rain_batch, target_batch in loader:
                water_batch = water_batch.to(self.device)
                rain_batch = rain_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                output = self.model(water_batch, rain_batch)
                loss = self.criterion(output, target_batch)
                losses.append(loss.item())
                all_preds.append(output.cpu().numpy())
                all_targets.append(target_batch.cpu().numpy())
        avg_loss = np.mean(losses)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        return avg_loss, all_preds, all_targets

    # 训练方法，执行训练过程并保存最佳模型
    def train(self, num_epochs, model_path):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for water_batch, rain_batch, target_batch in self.train_loader:
                water_batch = water_batch.to(self.device)
                rain_batch = rain_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(water_batch, rain_batch)
                loss = self.criterion(output, target_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(self.train_loader)
            val_loss, _, _ = self.evaluate(self.val_loader)
            print(f"[DoubleEncoder] Epoch {epoch+1}/50, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved best DoubleEncoder model to {model_path}")

class DoubleEncoderDataset(torch.utils.data.Dataset):
    # DoubleEncoder数据集，用于加载水、雨和目标值数据
    def __init__(self, water, rain, targets):
        self.water = water
        self.rain = rain
        self.targets = targets

    # 获取数据集的长度
    def __len__(self):
        return len(self.targets)

    # 获取指定索引的数据
    def __getitem__(self, idx):
        return self.water[idx], self.rain[idx], self.targets[idx]
