import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        # To ensure causality, we pad only the left side by (kernel_size - 1) * dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
        
    def forward(self, x):
        # x is (B, C, L)
        x = self.conv(x)
        # Because padding adds to both sides in standard Conv1d if padding is int, 
        # wait! PyTorch Conv1d padding adds symmetrically. 
        # Actually, if we pass padding to Conv1d, it adds it to BOTH sides!
        # To do asymmetric padding, we should pad manually and set Conv1d padding=0.
        pass

# Let's fix CausalConv1d properly
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        
    def forward(self, x):
        # x is (B, C, L)
        # Pad only the left side with zeros
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class FuelCausalTCN(nn.Module):
    def __init__(self):
        super(FuelCausalTCN, self).__init__()
        
        # Architecture approved by Reviewer
        # Layer 1: Causal Conv1D, 32 filters, kernel=3, dilation=1
        self.conv1 = CausalConv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=1)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Causal Dilated Conv1D, 64 filters, kernel=3, dilation=2
        self.conv2 = CausalConv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=2)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Causal Dilated Conv1D, 64 filters, kernel=3, dilation=4
        self.conv3 = CausalConv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=4)
        self.relu3 = nn.ReLU()
        
        # Dense Layers
        self.dense1 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.dense2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape expected from dataloader: (Batch, SeqLen=30, Features=5)
        # PyTorch Conv1d expects (Batch, Channels, SeqLen)
        x = x.transpose(1, 2) # Now (Batch, 5, 30)
        
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # Lấy timestep cuối cùng
        x_last = x[:, :, -1] # (Batch, 64)
        
        # Dense Network
        out = self.relu4(self.dense1(x_last))
        out = self.dense2(out) # (Batch, 1)
        
        # Return predicted residual
        # Note: Predicted FuelLevel = Predicted Residual + Anchor is done outside the network during inference
        return out.squeeze(-1)

class FuelCausalTCN_31(nn.Module):
    def __init__(self, in_channels=5):
        super(FuelCausalTCN_31, self).__init__()
        
        # Layer 1: RF += 2 -> Total RF = 3
        self.conv1 = CausalConv1d(in_channels=in_channels, out_channels=32, kernel_size=3, dilation=1)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Causal Dilated Conv1D, 64 filters, kernel=3, dilation=2
        self.conv2 = CausalConv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=2)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Causal Dilated Conv1D, 64 filters, kernel=3, dilation=4
        self.conv3 = CausalConv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=4)
        self.relu3 = nn.ReLU()
        
        # Layer 4: Causal Dilated Conv1D, 64 filters, kernel=3, dilation=8
        self.conv4 = CausalConv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=8)
        self.relu4 = nn.ReLU()
        
        # Dense Layers
        self.dense1 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()
        self.dense2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.transpose(1, 2) # (Batch, 5, 30)
        
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        x_last = x[:, :, -1] # (Batch, 64)
        
        out = self.relu5(self.dense1(x_last))
        out = self.dense2(out) # (Batch, 1)
        return out.squeeze(-1)
