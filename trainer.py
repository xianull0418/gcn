# trainer.py
import torch
import torch.optim as optim
import numpy as np

class SimpleTrainer:
    # SimpleTrainer类，负责训练和评估模型
    def __init__(self, model, train_loader, val_loader, lr=1e-4, device='cpu'):
        self.model = model.to(device)  # 将模型移动到指定设备
        self.train_loader = train_loader  # 训练数据加载器
        self.val_loader = val_loader  # 验证数据加载器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Adam优化器
        self.criterion = torch.nn.MSELoss()  # 损失函数，均方误差损失
        self.device = device  # 设备（cpu或gpu）

    # 评估方法，计算模型在给定数据加载器上的损失
    def evaluate(self, loader):
        self.model.eval()  # 设置模型为评估模式
        losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():  
            for x_batch, target_batch in loader:
                x_batch = x_batch.to(self.device)  
                target_batch = target_batch.to(self.device)  
                output = self.model(x_batch)  
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
            for x_batch, target_batch in self.train_loader:
                x_batch = x_batch.to(self.device)  
                target_batch = target_batch.to(self.device)  
                self.optimizer.zero_grad()  
                output = self.model(x_batch)  
                loss = self.criterion(output, target_batch) 
                loss.backward()  
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  
                self.optimizer.step()  
                epoch_loss += loss.item()  
            train_loss = epoch_loss / len(self.train_loader) 
            val_loss, _, _ = self.evaluate(self.val_loader)  
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
            if val_loss < best_val_loss:  
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_path) 
                print(f"保存了最佳模型到 {model_path}")
