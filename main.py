import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.common import DataProcessor, MergedWaterRainDataset, compute_rmse, compute_nse, compute_mae
from models.double_encoder_transformer import DoubleEncoderTransformer as DETModel, DoubleEncoderTrainer, DoubleEncoderDataset
from trainer import SimpleTrainer  # for merged-input models

def main(water_file, rain_file):
    # 根据water_file推导出数据特定的文件夹名称
    data_name = water_file.split("_")[1].split(".")[0]
    model_dir = os.path.join("./models", data_name)
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用的设备:", device)

    # 创建一个DataProcessor实例
    processor = DataProcessor(water_file, rain_file, date_col='TM',
                              water_val_col='Z', rain_val_col='DRP',
                              window_size=72, K_wl=9, K_rf=8)
    # 不直接调用processor.process()，我们还需要日期信息
    df = processor.load_and_merge_data()
    df = processor.scale_data(df)
    wl_imfs, rf_imfs = processor.apply_vmd(df)
    X_water, X_rain, Y = processor.create_sliding_windows(wl_imfs, rf_imfs)
    # 对应每个样本的日期从window_size索引开始
    all_dates = df['date'].iloc[processor.window_size:].reset_index(drop=True)

    # 对于双编码器Transformer，保持水和雨数据分开
    # 对于合并输入的模型，沿通道维度合并
    X_merged = torch.cat((X_water, X_rain), dim=1)

    # 数据集划分（70%训练，15%验证，15%测试）
    total_samples = X_merged.shape[0]
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)
    # 对于合并输入模型:
    X_train_merged = X_merged[:train_end]
    X_val_merged = X_merged[train_end:val_end]
    X_test_merged = X_merged[val_end:]
    Y_train = Y[:train_end]
    Y_val = Y[train_end:val_end]
    Y_test = Y[val_end:]
    test_dates = all_dates.iloc[val_end:].reset_index(drop=True)

    # 为合并输入的模型创建数据集和数据加载器
    train_dataset = MergedWaterRainDataset(X_train_merged, Y_train)
    val_dataset = MergedWaterRainDataset(X_val_merged, Y_val)
    test_dataset = MergedWaterRainDataset(X_test_merged, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 1. 双编码器Transformer（使用分别的水和雨输入）
    from models.double_encoder_transformer import DoubleEncoderTransformer as DETModel
    det_model = DETModel(d_model=64, nhead=4, K_wl=processor.K_wl, K_rf=processor.K_rf).to(device)
    det_model_path = os.path.join(model_dir, "DoubleEncoderTransformer_best.pth")
    # 为双编码器分别处理水和雨
    Xw_train = X_water[:train_end]
    Xw_val = X_water[train_end:val_end]
    Xw_test = X_water[val_end:]
    Xr_train = X_rain[:train_end]
    Xr_val = X_rain[train_end:val_end]
    Xr_test = X_rain[val_end:]

    train_dataset_det = DoubleEncoderDataset(Xw_train, Xr_train, Y_train)
    val_dataset_det = DoubleEncoderDataset(Xw_val, Xr_val, Y_val)
    test_dataset_det = DoubleEncoderDataset(Xw_test, Xr_test, Y_test)
    train_loader_det = DataLoader(train_dataset_det, batch_size=32, shuffle=True)
    val_loader_det = DataLoader(val_dataset_det, batch_size=32, shuffle=False)

    det_trainer = DoubleEncoderTrainer(det_model, train_loader_det, val_loader_det, lr=1e-4, device=device)
    det_trainer.train(num_epochs=50, model_path=det_model_path)
    _, det_preds, det_targets = det_trainer.evaluate(DataLoader(test_dataset_det, batch_size=32, shuffle=False))
    det_rmse = compute_rmse(det_targets, det_preds)
    det_nse = compute_nse(det_targets, det_preds)
    det_mae = compute_mae(det_targets, det_preds)
    print("双编码器Transformer模型结果:")
    print(f"RMSE: {det_rmse:.4f}, NSE: {det_nse:.4f}, MAE: {det_mae:.4f}")

    # 2. 其他合并输入模型（Transformer、LSTM、GRU、CNN）
    input_channels = X_merged.shape[1]
    window_size = processor.window_size
    from models.transformer_model import TransformerModel
    from models.lstm_model import LSTMModel
    from models.gru_model import GRUModel
    from models.cnn_model import CNNModel

    merged_models = {
        "Transformer": TransformerModel(d_model=64, nhead=4, num_layers=2,
                                        input_channels=input_channels, window_size=window_size),
        "LSTM": LSTMModel(hidden_dim=64, num_layers=2, input_channels=input_channels),
        "GRU": GRUModel(hidden_dim=64, num_layers=2, input_channels=input_channels),
        "CNN": CNNModel(input_channels=input_channels, window_size=window_size, num_filters=64, kernel_size=3)
    }
    merged_predictions = {}  # 存储每个模型的预测
    merged_results = {}
    for name, model in merged_models.items():
        print(f"\n训练{name}模型...")
        model_path = os.path.join(model_dir, f"{name}_best.pth")
        trainer = SimpleTrainer(model, train_loader, val_loader, lr=1e-4, device=device)
        trainer.train(num_epochs=50, model_path=model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, preds, targets = trainer.evaluate(DataLoader(test_dataset, batch_size=32, shuffle=False))
        rmse = compute_rmse(targets, preds)
        nse = compute_nse(targets, preds)
        mae = compute_mae(targets, preds)
        merged_results[name] = {"RMSE": rmse, "NSE": nse, "MAE": mae}
        merged_predictions[name] = preds  # 存储预测值
        print(f"--- {name} 结果 ---")
        print(f"RMSE: {rmse:.8f}, NSE: {nse:.8f}, MAE: {mae:.8f}")

    # 构建一个包含测试日期、实际值和每个模型预测值的DataFrame
    # 对于合并输入模型，实际目标值是相同的
    df_results = pd.DataFrame({
        "TM": test_dates.dt.strftime("%Y-%m-%d %H:%M:%S"),   # 按照指定格式格式化日期
        "Actual": Y_test.squeeze().numpy()
    })
    # 添加双编码器Transformer的预测（转换为1D numpy数组）
    df_results["DoubleEncoderTransformer"] = det_preds.squeeze()
    # 添加其他合并模型的预测值
    for name, preds in merged_predictions.items():
        df_results[name] = preds.squeeze()

    # 保存结果到CSV文件
    result_csv_path = os.path.join(model_dir, f"{data_name}_result.csv")
    df_results.to_csv(result_csv_path, index=False)
    print(f"\n所有测试预测值和实际值已保存到 {result_csv_path}")

    # 最后，打印所有模型的评估指标总结
    print("\n所有模型的总结:")
    print("双编码器Transformer:", {"RMSE": det_rmse, "NSE": det_nse, "MAE": det_mae})
    for name, metrics in merged_results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    water_file = "水位_东山.csv"
    rain_file = "雨量_东山.csv"
    main(water_file, rain_file)
