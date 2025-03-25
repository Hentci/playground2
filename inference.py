import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm

# 重用原始代码中的数据集类和模型类
class ScatterPlotDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, response_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        # 加载响应数据
        self.response_df = pd.read_csv(response_file)
        
        # 筛选存在的图像
        self.valid_indices = []
        self.image_ids = []
        
        for i, row in self.response_df.iterrows():
            image_id = row['id']
            image_path = os.path.join(images_dir, f"{image_id}.png")
            
            if os.path.exists(image_path):
                self.valid_indices.append(i)
                self.image_ids.append(image_id)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        row_idx = self.valid_indices[idx]
        row = self.response_df.iloc[row_idx]
        
        image_id = row['id']
        correlation = float(row['corr'])
        
        image_path = os.path.join(self.images_dir, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, correlation, image_id

# CNN模型用于相关性预测
class CorrelationCNN(nn.Module):
    def __init__(self):
        super(CorrelationCNN, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8 -> 4x4
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def inference(model_path, data_loader, device):
    """
    使用加载的模型对测试集进行推理
    """
    # 加载模型
    model = CorrelationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 存储结果
    predictions = []
    true_values = []
    image_ids = []
    
    # 进行推理
    with torch.no_grad():
        for inputs, targets, ids in tqdm(data_loader, desc="Inferencing"):
            inputs = inputs.to(device)
            targets = targets.numpy()
            
            # 模型预测
            outputs = model(inputs).cpu().numpy()
            
            # 存储结果
            predictions.extend(outputs.flatten())
            true_values.extend(targets)
            image_ids.extend(ids)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'true_correlation': true_values,
        'predicted_correlation': predictions
    })
    
    # 计算误差指标
    results_df['absolute_error'] = np.abs(results_df['true_correlation'] - results_df['predicted_correlation'])
    results_df['squared_error'] = (results_df['true_correlation'] - results_df['predicted_correlation']) ** 2
    
    return results_df

def evaluate_model(results_df):
    """
    评估模型性能并输出各种指标
    """
    # 计算各种指标
    mae = results_df['absolute_error'].mean()
    mse = results_df['squared_error'].mean()
    rmse = np.sqrt(mse)
    max_error = results_df['absolute_error'].max()
    
    # 输出结果
    print("\n===== Model Evaluation Results =====")
    print(f"Test set samples: {len(results_df)}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Maximum Absolute Error: {max_error:.6f}")
    
    # 找出误差最大的预测
    worst_predictions = results_df.sort_values('absolute_error', ascending=False).head(10)
    print("\n===== Top 10 Samples with Largest Prediction Errors =====")
    print(worst_predictions[['image_id', 'true_correlation', 'predicted_correlation', 'absolute_error']])
    
    return mae, mse, rmse

def visualize_results(results_df, output_dir):
    """
    可视化预测结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 真实值与预测值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['true_correlation'], results_df['predicted_correlation'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # 完美预测线
    plt.xlabel('Ground Truth Correlation')
    plt.ylabel('Predicted Correlation')
    plt.title('Ground Truth vs Predicted Correlation')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
    
    # 2. 误差分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['absolute_error'], bins=50, alpha=0.75)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    
    # 3. 真实值与误差散点图（查看是否有特定区域的系统性误差）
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['true_correlation'], results_df['absolute_error'], alpha=0.5)
    plt.xlabel('Ground Truth Correlation')
    plt.ylabel('Absolute Error')
    plt.title('Ground Truth vs Prediction Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_vs_true.png'))
    
    # 关闭所有图形
    plt.close('all')

def predict_single_image(model_path, image_path, device):
    """
    预测单张图像的相关性
    """
    # 加载模型
    model = CorrelationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        prediction = model(image_tensor).item()
    
    return prediction

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Scatter Plot Correlation Prediction - Inference Code')
    parser.add_argument('--model_path', type=str, default='directly_train_log/best_model.pth', help='Path to trained model')
    parser.add_argument('--images_dir', type=str, default='correlation_assignment/images', help='Directory containing images')
    parser.add_argument('--response_file', type=str, default='correlation_assignment/responses.csv', help='CSV file with ground truth correlations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory for output results')
    parser.add_argument('--single_image', type=str, default=None, help='Path to single image (optional)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 单张图像预测模式
    if args.single_image is not None:
        if not os.path.exists(args.single_image):
            print(f"Error: Image {args.single_image} does not exist")
            return
            
        prediction = predict_single_image(args.model_path, args.single_image, device)
        print(f"\nPredicted correlation for image {os.path.basename(args.single_image)}: {prediction:.6f}")
        return
    
    # 批量推理模式
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
        
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    print("Loading data...")
    dataset = ScatterPlotDataset(args.images_dir, args.response_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} images with correlation values")
    
    # 执行推理
    print("\nStarting inference...")
    results_df = inference(args.model_path, data_loader, device)
    
    # 保存原始结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'inference_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # 评估模型
    mae, mse, rmse = evaluate_model(results_df)
    
    # 可视化结果
    print("\nGenerating visualizations...")
    visualize_results(results_df, args.output_dir)
    print(f"Visualization results saved to {args.output_dir} directory")

if __name__ == "__main__":
    main()