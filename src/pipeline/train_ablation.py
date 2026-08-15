import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.time_aware_gru import FuelTimeAwareGRU

def train_ablation_model(name, loss_type, weight_decay, lambda_smooth=0.0, N=30, epochs=25, batch_size=256, patience=4):
    print(f"\n{'='*50}\nTraining Ablation {name} (Loss: {loss_type}, WD: {weight_decay}, CTS_lambda: {lambda_smooth})\n{'='*50}")
    
    data_dir = 'data/gru_dataset/windows'
    X_train = np.load(os.path.join(data_dir, f'X_train_N{N}.npy'))
    y_train = np.load(os.path.join(data_dir, f'y_train_N{N}.npy'))
    mask_train = np.load(os.path.join(data_dir, f'mask_train_N{N}.npy'))
    
    X_val = np.load(os.path.join(data_dir, f'X_val_N{N}.npy'))
    y_val = np.load(os.path.join(data_dir, f'y_val_N{N}.npy'))
    mask_val = np.load(os.path.join(data_dir, f'mask_val_N{N}.npy'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(mask_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val), torch.tensor(mask_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = FuelTimeAwareGRU(input_dim=6, hidden_dim=64).to(device)
    
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=5.0)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('models/gru', exist_ok=True)
    model_path = f'models/gru/best_gru_ablation_{name}.pth'
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, mask_batch in train_loader:
            X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Conditional Temporal Smoothness (CTS) Loss
            if lambda_smooth > 0.0:
                # Get predictions for t-1 and t-2 by truncating the sequence
                # X_batch[:, :-1, :] -> predictions at t-1
                # X_batch[:, :-2, :] -> predictions at t-2
                outputs_t1 = model(X_batch[:, :-1, :])
                outputs_t2 = model(X_batch[:, :-2, :])
                
                # Second difference: (y_hat_t - 2*y_hat_{t-1} + y_hat_{t-2})
                sec_diff = outputs - 2 * outputs_t1 + outputs_t2
                
                # Mask specifies 1 for Normal/Sloshing and 0 for Refuel/Drain
                # Only apply smoothness where mask == 1
                smoothness_loss = torch.mean(mask_batch * (sec_diff ** 2))
                
                loss += lambda_smooth * smoothness_loss
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                if lambda_smooth > 0.0:
                    outputs_t1 = model(X_batch[:, :-1, :])
                    outputs_t2 = model(X_batch[:, :-2, :])
                    sec_diff = outputs - 2 * outputs_t1 + outputs_t2
                    smoothness_loss = torch.mean(mask_batch * (sec_diff ** 2))
                    loss += lambda_smooth * smoothness_loss
                    
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
                
    print(f"Finished training {name}. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

if __name__ == '__main__':
    # Only train Model E
    experiments = [
        ('E', 'MSE', 1e-4, 10.0), # lambda = 10.0 to strongly enforce smoothness
    ]
    
    results = {}
    for name, loss_type, wd, lam in experiments:
        val_loss = train_ablation_model(name, loss_type, wd, lambda_smooth=lam, epochs=25, patience=4)
        results[name] = val_loss
        
    print("\n=== ABLATION RESULTS (Val Loss is metric-dependent) ===")
    for name, loss in results.items():
        print(f"Model {name}: Best Val Loss = {loss:.4f}")
