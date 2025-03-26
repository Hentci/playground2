import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom dataset for scatter plots
class ScatterPlotDataset(Dataset):
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
            nn.Linear(128 * 4 * 4, 512),  # 4x4 feature maps after pooling
            nn.ReLU(),
            nn.Linear(512, 1)  # No activation for regression output
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=10):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_mae += torch.sum(torch.abs(outputs - targets)).item()
        
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_mae += torch.sum(torch.abs(outputs - targets)).item()
            
            val_loss /= len(val_loader.dataset)
            val_mae /= len(val_loader.dataset)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Train MAE: {train_mae:.4f} | '
              f'Val MAE: {val_mae:.4f}')
        
        # Check if we need to save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
        
        # Early stopping
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

# Function to identify outliers and calculate test MAE
def analyze_results(model, test_loader, device):
    model.eval()
    
    y_true = []
    y_pred = []
    image_ids = []
    
    with torch.no_grad():
        for inputs, targets, ids in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            
            outputs = model(inputs)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            image_ids.extend(ids)
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate errors
    abs_errors = np.abs(y_pred - y_true)
    
    # Calculate Test MAE
    test_mae = np.mean(abs_errors)
    print(f'\nTest MAE: {test_mae:.4f}')
    
    # Calculate Test MSE
    test_mse = np.mean(np.square(y_pred - y_true))
    print(f'Test MSE: {test_mse:.4f}')
    
    # Identify outliers (using statistical threshold)
    threshold = np.mean(abs_errors) + 2 * np.std(abs_errors)
    is_outlier = abs_errors > threshold
    
    print(f'Number of potential outliers: {sum(is_outlier)}')
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'true_correlation': y_true,
        'predicted_correlation': y_pred,
        'absolute_error': abs_errors,
        'is_outlier': is_outlier
    })
    
    # Sort by error
    sorted_results = results_df.sort_values('absolute_error', ascending=False)
    print("\nTop 10 worst predictions:")
    print(sorted_results.head(10))
    
    results_df.to_csv('prediction_results.csv', index=False)
    
    return results_df, test_mae, test_mse

def main():
    # Set base directory
    base_dir = 'data'
    
    # Set paths for train, test, valid directories
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    valid_dir = os.path.join(base_dir, 'valid')
    
    # Set path for responses.csv
    response_file = os.path.join(base_dir, 'responses.csv')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Image preprocessing - resize to 64x64 as requested
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets for each split
    print("Loading data...")
    train_dataset = ScatterPlotDataset(train_dir, response_file, transform=transform)
    valid_dataset = ScatterPlotDataset(valid_dir, response_file, transform=transform)
    test_dataset = ScatterPlotDataset(test_dir, response_file, transform=transform)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = CorrelationCNN().to(device)
    
    # Define loss function (MSE) and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("\nTraining model...")
    model, history = train_model(
        model, train_loader, valid_loader, criterion, optimizer, device,
        num_epochs=50, patience=10
    )
    
    # Analyze results and identify potential outliers
    print("\nAnalyzing results...")
    results_df, test_mae, test_mse = analyze_results(model, test_loader, device)
    
    # 將測試指標加入到history字典中，以便可以與訓練和驗證指標一起保存
    history['test_mae'] = test_mae
    history['test_mse'] = test_mse
    
    # 儲存所有指標到CSV檔案
    metrics_df = pd.DataFrame({
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'test_mse': test_mse,
        'final_train_mae': history['train_mae'][-1],
        'final_val_mae': history['val_mae'][-1],
        'test_mae': test_mae
    }, index=[0])
    
    metrics_df.to_csv('model_metrics.csv', index=False)
    print(f"\nFinal metrics saved to model_metrics.csv")
    print(f"Final Test MAE: {test_mae:.4f}")
    print(f"Final Test MSE: {test_mse:.4f}")
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'])
    plt.plot(history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Plot actual vs predicted correlations
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['true_correlation'], results_df['predicted_correlation'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Correlation')
    plt.ylabel('Predicted Correlation')
    plt.title('Actual vs Predicted Correlation Values')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig('prediction_scatter.png')

if __name__ == "__main__":
    main()