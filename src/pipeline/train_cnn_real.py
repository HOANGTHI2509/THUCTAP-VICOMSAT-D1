import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.cnn_gated_attention import FuelCNN1DGatedAttention

def train_cnn_model(N=30, epochs=50, batch_size=256, patience=10, lr=1e-3, weight_decay=1e-4):
    print(f"\n{'='*50}\nTraining CNN-Gated Attention on Real Data (N={N}, Loss: Huber)\n{'='*50}")
    
    data_dir = 'data/real_dataset/windows'
    
    if not os.path.exists(os.path.join(data_dir, f'X_train_N{N}.npy')):
        print(f"Dataset for N={N} not found in {data_dir}. Please run build_real_dataset.py first.")
        return
        
    print("Loading real data...")
    X_train = np.load(os.path.join(data_dir, f'X_train_N{N}.npy'))
    y_train = np.load(os.path.join(data_dir, f'y_train_N{N}.npy'))
    
    X_val = np.load(os.path.join(data_dir, f'X_val_N{N}.npy'))
    y_val = np.load(os.path.join(data_dir, f'y_val_N{N}.npy'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = 6
    hidden_dim = 64
    model = FuelCNN1DGatedAttention(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Criterion will be applied custom with weights inside the loop
    # criterion = nn.HuberLoss(delta=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    out_dir = 'models/cnn_real'
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model_real.pt')
    
    history = {'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            base_loss = F.huber_loss(outputs.squeeze(), y_batch.squeeze(), delta=0.05, reduction='none')
            magnitude = torch.abs(y_batch.squeeze())
            weight = 1.0 + 10.0 * torch.clamp(magnitude / 0.02, max=1.0)
            loss = torch.mean(weight * base_loss)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                base_loss = F.huber_loss(outputs.squeeze(), y_batch.squeeze(), delta=0.05, reduction='none')
                magnitude = torch.abs(y_batch.squeeze())
                weight = 1.0 + 10.0 * torch.clamp(magnitude / 0.02, max=1.0)
                loss = torch.mean(weight * base_loss)
                
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f" -> Saved new best model (Val Loss: {best_val_loss:.8f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
                
    train_time = time.time() - start_time
    print(f"\nFinished training in {train_time:.2f} seconds. Best Val Loss: {best_val_loss:.8f}")
    
    config = {
        'model': '1D-CNN + Gated Attention',
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'kernel_size': 3,
        'window_size': N,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'loss_function': 'HuberLoss(delta=0.05)',
        'best_val_loss': best_val_loss,
        'total_params': total_params,
        'train_time_seconds': train_time
    }
    
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    np.savetxt(os.path.join(out_dir, 'training_history.csv'), 
               np.column_stack((history['train_loss'], history['val_loss'])), 
               delimiter=',', header='train_loss,val_loss', comments='')
               
    print(f"Artifacts saved to {out_dir}/")

if __name__ == '__main__':
    train_cnn_model(N=10)
