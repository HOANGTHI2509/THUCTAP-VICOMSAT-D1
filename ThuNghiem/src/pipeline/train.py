import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.tcn_model import FuelCausalTCN_31

def load_train_val_data():
    base_path = "data/tcn_dataset/"
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    
    X_val = np.load(os.path.join(base_path, "X_val.npy"))
    y_val = np.load(os.path.join(base_path, "y_val.npy"))
    
    return X_train, y_train, X_val, y_val

def train_ablation_model(model, train_loader, val_loader, model_name, device, max_epochs=20):
    print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
    
    criterion = nn.HuberLoss(delta=10.0) # Delta 10.0 for jump protection
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    patience = 5
    
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            
            criterion = nn.HuberLoss(delta=10.0, reduction='none')
            
            # y_target_array has [y_std_target, y_ep_target, weight]
            # y_ep_target is at index 1, weight is at index 2
            targets = batch_y[:, 1]
            weights = batch_y[:, 2]
            
            loss_unweighted = criterion(preds, targets)
            loss = (loss_unweighted * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        model.eval()
        criterion = nn.HuberLoss(delta=10.0, reduction='none')
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                targets = batch_y[:, 1]
                weights = batch_y[:, 2]
                
                loss_unweighted = criterion(preds, targets)
                loss = (loss_unweighted * weights).mean()
                val_loss_sum += loss.item()
                
        avg_val_loss = val_loss_sum / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"models/tcn_weights/{model_name}.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    return best_val_loss

def run_ablation():
    os.makedirs("models/tcn_weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train_full, y_train, X_val_full, y_val = load_train_val_data()
    batch_size = 1024
    
    ablation_plans = {
        'M1': [0],          # FuelResidual
        'M2': [0, 1],       # FuelResidual + Speed
        'M3': [0, 1, 2],    # FuelResidual + Speed + Accel
        'M4': [0, 1, 2, 3, 4] # All features
    }
    
    for model_name, feature_indices in ablation_plans.items():
        print(f"\nPreparing data for {model_name} with features: {feature_indices}")
        
        X_train_sub = X_train_full[:, :, feature_indices]
        X_val_sub = X_val_full[:, :, feature_indices]
        
        train_ds = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val_sub, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        model = FuelCausalTCN_31(in_channels=len(feature_indices)).to(device)
        train_ablation_model(model, train_loader, val_loader, model_name, device)

if __name__ == "__main__":
    run_ablation()
