import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    def __init__(self, d_model):
        super(GatedAttention, self).__init__()
        self.attn_W = nn.Linear(d_model, d_model)
        self.attn_U = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, SeqLen, d_model)
        a = torch.tanh(self.attn_W(x))
        b = torch.sigmoid(self.attn_U(x))
        
        # Element-wise multiplication
        gated = a * b
        
        # Calculate attention weights
        attn_weights = self.attn_v(gated) # (Batch, SeqLen, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        context = torch.sum(attn_weights * x, dim=1) # (Batch, d_model)
        return context, attn_weights

class FuelCNN1DGatedAttention(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super(FuelCNN1DGatedAttention, self).__init__()
        
        # Chuẩn hóa ngay tại đầu vào (chống lại sự chênh lệch tỷ lệ giữa các đặc trưng như Speed ~ 80, Residual ~ 5)
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # 1D CNN for local feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.relu = nn.ReLU()
        
        # Gated Attention
        self.attention = GatedAttention(hidden_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (Batch, SeqLen, Features)
        
        # Convert for Conv1d: (Batch, Features, SeqLen)
        x = x.permute(0, 2, 1)
        
        # Chuẩn hóa đầu vào
        x = self.input_bn(x)
        
        # Pass through CNN
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        # Convert back for Attention: (Batch, SeqLen, Features)
        out = out.permute(0, 2, 1)
        
        # Pass through Gated Attention
        context, attn_weights = self.attention(out)
        
        # Pass through dense layers
        out = self.relu(self.fc1(context))
        out = self.fc2(out)
        out = 0.05 * torch.tanh(out)
        
        # out shape: (Batch, 1)
        # The model directly predicts the residual (TrueFuel(t) - Anchor)
        return out
