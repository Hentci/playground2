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

# ScatterPlot dataset class
class ScatterPlotDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, response_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        # Load response data
        self.response_df = pd.read_csv(response_file)
        
        # Filter existing images
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

# CNN model for correlation prediction
class CorrelationCNN(nn.Module):
    def __init__(self):
        super(CorrelationCNN, self).__init__()
        
        # Convolutional layers
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
        
        # Fully connected layers
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
    Run inference on test data using the loaded model
    """
    # Load model
    model = CorrelationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Store results
    predictions = []
    true_values = []
    image_ids = []
    
    # Run inference
    with torch.no_grad():
        for inputs, targets, ids in tqdm(data_loader, desc="Running inference"):
            inputs = inputs.to(device)
            targets = targets.numpy()
            
            # Model prediction
            outputs = model(inputs).cpu().numpy()
            
            # Store results
            predictions.extend(outputs.flatten())
            true_values.extend(targets)
            image_ids.extend(ids)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'true_correlation': true_values,
        'predicted_correlation': predictions
    })
    
    # Calculate error metrics
    results_df['absolute_error'] = np.abs(results_df['true_correlation'] - results_df['predicted_correlation'])
    results_df['squared_error'] = (results_df['true_correlation'] - results_df['predicted_correlation']) ** 2
    
    return results_df

def evaluate_model(results_df):
    """
    Evaluate model performance and output various metrics
    """
    # Calculate metrics
    mae = results_df['absolute_error'].mean()
    mse = results_df['squared_error'].mean()
    rmse = np.sqrt(mse)
    max_error = results_df['absolute_error'].max()
    
    # Print results
    print("\n===== Model Evaluation Results =====")
    print(f"Test set samples: {len(results_df)}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Maximum Absolute Error: {max_error:.6f}")
    
    # Find worst predictions
    worst_predictions = results_df.sort_values('absolute_error', ascending=False).head(10)
    print("\n===== Top 10 Samples with Largest Prediction Errors =====")
    print(worst_predictions[['image_id', 'true_correlation', 'predicted_correlation', 'absolute_error']])
    
    return mae, mse, rmse

def visualize_results(results_df, output_dir):
    """
    Visualize prediction results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ground truth vs predicted scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['true_correlation'], results_df['predicted_correlation'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Ground Truth Correlation')
    plt.ylabel('Predicted Correlation')
    plt.title('Ground Truth vs Predicted Correlation')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
    
    # 2. Error distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['absolute_error'], bins=50, alpha=0.75)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    
    # 3. Ground truth vs error scatter plot (check for systematic errors in specific regions)
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['true_correlation'], results_df['absolute_error'], alpha=0.5)
    plt.xlabel('Ground Truth Correlation')
    plt.ylabel('Absolute Error')
    plt.title('Ground Truth vs Prediction Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_vs_true.png'))
    
    # Close all figures
    plt.close('all')

def predict_single_image(model_path, image_path, device):
    """
    Predict correlation for a single image
    """
    # Load model
    model = CorrelationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(image_tensor).item()
    
    return prediction

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Scatter Plot Correlation Prediction - Inference Code')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Base directory containing data folders')
    parser.add_argument('--test_dir', type=str, default='test', help='Test directory name within data_dir')
    parser.add_argument('--response_file', type=str, default='responses.csv', help='CSV file with ground truth correlations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory for output results')
    parser.add_argument('--single_image', type=str, default=None, help='Path to single image (optional)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set up paths
    base_dir = args.data_dir
    test_dir = os.path.join(base_dir, args.test_dir)
    response_file = os.path.join(base_dir, args.response_file)
    
    # Single image prediction mode
    if args.single_image is not None:
        if not os.path.exists(args.single_image):
            print(f"Error: Image {args.single_image} does not exist")
            return
            
        prediction = predict_single_image(args.model_path, args.single_image, device)
        print(f"\nPredicted correlation for image {os.path.basename(args.single_image)}: {prediction:.6f}")
        return
    
    # Batch inference mode
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
        
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and data loader
    print("Loading data...")
    dataset = ScatterPlotDataset(test_dir, response_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} images with correlation values from test set")
    
    # Run inference
    print("\nStarting inference...")
    results_df = inference(args.model_path, data_loader, device)
    
    # Save raw results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'inference_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Evaluate model
    mae, mse, rmse = evaluate_model(results_df)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results_df, args.output_dir)
    print(f"Visualization results saved to {args.output_dir} directory")

if __name__ == "__main__":
    main()