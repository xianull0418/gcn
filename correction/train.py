import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from correction.config import Config
from correction.models.stgcn import STGCN
from correction.utils.data_processor import DataProcessor
from correction.utils.graph_builder import GraphBuilder
from correction.utils.metrics import Metrics
from correction.optimization.optimizer import ModelOptimizer
from correction.models.baseline_models import BaselineModels

class CorrectionTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.setup_paths()
        
    def setup_paths(self):
        """设置必要的路径"""
        # 确保所需目录存在
        Path(self.config.PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.config.MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self):
        """准备训练数据"""
        # 加载和处理数据
        processor = DataProcessor(self.config)
        features, labels, water_bins = processor.process_data()
        
        # 构建图结构
        graph_builder = GraphBuilder(self.config)
        adj_matrix = graph_builder.build_water_level_graph(water_bins)
        graph_features = graph_builder.build_dynamic_graph(features, water_bins)
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(graph_features)
        labels_tensor = torch.FloatTensor(labels)
        adj_tensor = adj_matrix.to(self.device)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(features_tensor))
        train_features = features_tensor[:train_size]
        train_labels = labels_tensor[:train_size]
        val_features = features_tensor[train_size:]
        val_labels = labels_tensor[train_size:]
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            train_features, 
            adj_tensor.unsqueeze(0).expand(len(train_features), -1, -1),  # 使用expand而不是repeat
            train_labels
        )
        val_dataset = TensorDataset(
            val_features,
            adj_tensor.unsqueeze(0).expand(len(val_features), -1, -1),
            val_labels
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE
        )
        
        return train_loader, val_loader, adj_tensor
    
    def train_epoch(self, model, train_loader, optimizer, criterion, adj):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        
        for batch_features, batch_adj, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_adj = batch_adj.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_features, batch_adj)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, model, val_loader, criterion, adj):
        """验证模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_adj, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_adj = batch_adj.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_features, batch_adj)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                all_preds.append(outputs.cpu())
                all_labels.append(batch_labels.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = Metrics.calculate_metrics(all_labels, all_preds)
        
        return total_loss / len(val_loader), metrics
    
    def train_baseline_models(self, train_loader, val_loader):
        """训练和评估基准模型"""
        # 收集数据
        train_features = []
        train_labels = []
        val_features = []
        val_labels = []
        
        for features, labels in train_loader:
            train_features.append(features.cpu().numpy())
            train_labels.append(labels.cpu().numpy())
        
        for features, labels in val_loader:
            val_features.append(features.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
        
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        val_features = np.concatenate(val_features, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        
        # 训练和评估基准模型
        baseline = BaselineModels(self.config)
        baseline_results = baseline.train_and_evaluate(
            train_features,
            train_labels,
            val_features,
            val_labels
        )
        
        return baseline_results
    
    def train(self):
        """执行完整的训练流程"""
        # 准备数据
        train_loader, val_loader, adj = self.prepare_data()
        
        # 初始化模型
        model = STGCN(self.config).to(self.device)
        
        # 优化超参数
        print("开始超参数优化...")
        model_optimizer = ModelOptimizer(
            self.config, 
            model, 
            train_loader, 
            val_loader
        )
        best_params, best_fitness = model_optimizer.optimize()
        print(f"最佳超参数: {best_params}")
        print(f"最佳适应度: {best_fitness}")
        
        # 使用优化后的参数重新初始化模型
        model = STGCN(self.config).to(self.device)
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE
        )
        criterion = torch.nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                adj
            )
            val_loss, metrics = self.validate(
                model, 
                val_loader, 
                criterion, 
                adj
            )
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            Metrics.print_metrics(metrics)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 
                         f"{self.config.MODEL_SAVE_PATH}/best_model.pth")
                print("保存最佳模型")
            
            print("-" * 50)
        
        # 训练基准模型
        print("\n开始训练基准模型...")
        baseline_results = self.train_baseline_models(train_loader, val_loader)
        
        # 比较结果
        print("\n模型性能对比:")
        print("-" * 50)
        print("STGCN模型:")
        Metrics.print_metrics(metrics)  # 使用最后一个epoch的验证指标
        
        for name, result in baseline_results.items():
            print(f"\n{name} 模型:")
            Metrics.print_metrics(result['val_metrics'])

if __name__ == "__main__":
    config = Config()
    trainer = CorrectionTrainer(config)
    trainer.train() 