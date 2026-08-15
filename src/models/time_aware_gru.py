import torch
import torch.nn as nn

class FuelTimeAwareGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1):
        super(FuelTimeAwareGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Time-aware GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (Batch, SequenceLength, Features)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # We only want the output from the last time step
        out = out[:, -1, :]
        
        # Pass through dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        # Extract the current noisy residual: NoisyFuel(t) - Anchor
        # x is shape (Batch, SeqLen, Features). Feature 0 is FuelResidual.
        current_noisy_res = x[:, -1, 0:1]
        
        # out predicts the correction (CleanFuel - NoisyFuel)
        # We add current_noisy_res to get (CleanFuel - Anchor)
        predicted_residual = out + current_noisy_res
        
        # out shape: (Batch, 1)
        return predicted_residual
