import torch
import numpy as np
from .pso_gwo import PSOGWO
from correction.models.stgcn import STGCN
from correction.utils.metrics import Metrics

class ModelOptimizer:
    def __init__(self, config, model, train_data, val_data):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        
        # 定义需要优化的超参数范围
        self.param_bounds = np.array([
            [32, 256],     # hidden_channels
            [1e-5, 1e-2],  # learning_rate
            [2, 6],        # num_layers
            [0.0, 0.5],    # dropout_rate
        ])
        
    def fitness_function(self, params):
        """评估一组超参数的适应度"""
        print(f"\n正在评估参数: {params}")
        
        try:
            # 更新模型超参数
            self.config.HIDDEN_CHANNELS = int(params[0])
            self.config.LEARNING_RATE = params[1]
            self.config.NUM_LAYERS = int(params[2])
            
            print(f"Hidden channels: {self.config.HIDDEN_CHANNELS}")
            print(f"Learning rate: {self.config.LEARNING_RATE}")
            print(f"Num layers: {self.config.NUM_LAYERS}")
            
            # 重新初始化模型
            model = STGCN(self.config).to(self.config.DEVICE)
            model.output_proj[-2].p = params[3]  # 更新dropout率
            print(f"Dropout rate: {params[3]}")
            
            # 训练模型
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
            criterion = torch.nn.MSELoss()
            
            # 简单训练几个epoch来评估参数效果
            model.train()
            train_losses = []
            for epoch in range(5):
                epoch_loss = 0
                batch_count = 0
                
                for batch_features, batch_adj, batch_labels in self.train_data:
                    try:
                        batch_features = batch_features.to(self.config.DEVICE)
                        batch_adj = batch_adj.to(self.config.DEVICE)
                        batch_labels = batch_labels.to(self.config.DEVICE)
                        
                        optimizer.zero_grad()
                        output = model(batch_features, batch_adj)
                        loss = criterion(output, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                        
                    except Exception as e:
                        print(f"批次处理错误: {str(e)}")
                        print(f"Batch shapes - features: {batch_features.shape}, adj: {batch_adj.shape}, labels: {batch_labels.shape}")
                        raise e
                
                avg_epoch_loss = epoch_loss / batch_count
                train_losses.append(avg_epoch_loss)
                print(f"Epoch {epoch+1}/5 - Loss: {avg_epoch_loss:.6f}")
            
            # 在验证集上评估
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_features, batch_adj, batch_labels in self.val_data:
                    batch_features = batch_features.to(self.config.DEVICE)
                    batch_adj = batch_adj.to(self.config.DEVICE)
                    batch_labels = batch_labels.to(self.config.DEVICE)
                    
                    output = model(batch_features, batch_adj)
                    val_loss = criterion(output, batch_labels)
                    val_losses.append(val_loss.item())
            
            mean_val_loss = np.mean(val_losses)
            print(f"验证集损失: {mean_val_loss:.6f}")
            return mean_val_loss
            
        except Exception as e:
            print(f"参数评估出错: {str(e)}")
            # 返回一个较大的损失值
            return 1e6
    
    def optimize(self):
        """执行超参数优化"""
        optimizer = PSOGWO(self.config, self.fitness_function)
        best_params, best_fitness = optimizer.optimize(
            dim=len(self.param_bounds),
            bounds=self.param_bounds
        )
        
        # 更新最佳超参数
        self.config.HIDDEN_CHANNELS = int(best_params[0])
        self.config.LEARNING_RATE = best_params[1]
        self.config.NUM_LAYERS = int(best_params[2])
        self.model.output_proj[-2].p = best_params[3]
        
        return best_params, best_fitness 