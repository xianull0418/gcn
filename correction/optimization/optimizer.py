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
        
        self.best_model = None
        self.eval_interval = 10  # 每10次迭代评估一次
        
    def fitness_function(self, params):
        """评估一组超参数的适应度"""
        try:
            # 更新模型超参数
            self.config.HIDDEN_CHANNELS = int(params[0])
            self.config.LEARNING_RATE = params[1]
            self.config.NUM_LAYERS = int(params[2])
            
            # 重新初始化模型
            model = STGCN(self.config).to(self.config.DEVICE)
            model.output_proj[-2].p = params[3]
            
            # 训练模型
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
            criterion = torch.nn.MSELoss()
            
            # 简化训练过程
            model.train()
            for epoch in range(self.config.EVAL_EPOCHS):
                for batch_features, batch_adj, batch_labels in self.train_data:
                    batch_features = batch_features.to(self.config.DEVICE)
                    batch_adj = batch_adj.to(self.config.DEVICE)
                    batch_labels = batch_labels.to(self.config.DEVICE)
                    
                    optimizer.zero_grad()
                    output = model(batch_features, batch_adj)
                    loss = criterion(output, batch_labels)
                    loss.backward()
                    optimizer.step()
            
            # 快速评估
            model.eval()
            val_loss = 0
            n_batches = 0
            with torch.no_grad():
                for batch_features, batch_adj, batch_labels in self.val_data:
                    batch_features = batch_features.to(self.config.DEVICE)
                    batch_adj = batch_adj.to(self.config.DEVICE)
                    batch_labels = batch_labels.to(self.config.DEVICE)
                    
                    output = model(batch_features, batch_adj)
                    val_loss += criterion(output, batch_labels).item()
                    n_batches += 1
                    
                    # 只评估部分验证集
                    if n_batches >= 5:
                        break
            
            return val_loss / n_batches
            
        except Exception as e:
            print(f"参数评估出错: {str(e)}")
            return 1e6
    
    def evaluate_correction(self, model):
        """评估校正效果"""
        model.eval()
        original_errors = []
        corrected_errors = []
        
        with torch.no_grad():
            for batch_features, batch_adj, batch_labels in self.val_data:
                batch_features = batch_features.to(self.config.DEVICE)
                batch_adj = batch_adj.to(self.config.DEVICE)
                batch_labels = batch_labels.to(self.config.DEVICE)
                
                # 获取校正值
                corrections = model(batch_features, batch_adj)
                
                # 收集原始误差和校正后的误差
                original_errors.append(batch_labels.cpu().numpy())
                corrected_errors.append((batch_labels - corrections).cpu().numpy())
        
        # 转换为numpy数组
        original_errors = np.concatenate(original_errors, axis=0)
        corrected_errors = np.concatenate(corrected_errors, axis=0)
        
        # 计算每个模型的RMSE改善程度
        improvements = {}
        model_names = ['DoubleEncoderTransformer', 'Transformer', 'LSTM', 'GRU', 'CNN']
        
        for i, name in enumerate(model_names):
            original_rmse = np.sqrt(np.mean(original_errors[:, i] ** 2))
            corrected_rmse = np.sqrt(np.mean(corrected_errors[:, i] ** 2))
            improvement = (original_rmse - corrected_rmse) / original_rmse * 100
            improvements[name] = improvement
        
        return improvements
    
    def optimize(self):
        """执行超参数优化"""
        optimizer = PSOGWO(self.config, self.fitness_function)
        iteration_count = 0
        
        def callback(best_solution, best_fitness):
            nonlocal iteration_count
            iteration_count += 1
            
            if iteration_count % self.eval_interval == 0:
                # 使用当前最佳参数创建模型
                model = STGCN(self.config).to(self.config.DEVICE)
                model.output_proj[-2].p = best_solution[3]
                
                # 评估校正效果
                improvements = self.evaluate_correction(model)
                
                print("\n" + "="*50)
                print(f"迭代 {iteration_count} 的校正效果:")
                for model_name, improvement in improvements.items():
                    print(f"{model_name}: 提升 {improvement:.2f}%")
                print("="*50 + "\n")
                
                # 保存最佳模型
                if self.best_model is None or best_fitness < self.best_fitness:
                    self.best_model = model.state_dict()
                    self.best_fitness = best_fitness
        
        best_params, best_fitness = optimizer.optimize(
            dim=len(self.param_bounds),
            bounds=self.param_bounds,
            callback=callback
        )
        
        # 更新最佳超参数
        self.config.HIDDEN_CHANNELS = int(best_params[0])
        self.config.LEARNING_RATE = best_params[1]
        self.config.NUM_LAYERS = int(best_params[2])
        self.model.output_proj[-2].p = best_params[3]
        
        # 加载最佳模型
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        return best_params, best_fitness 