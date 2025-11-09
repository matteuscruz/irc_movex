import torch.nn as nn
from .base_trainer import BaseTrainer

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=5):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class GRUTrainer(BaseTrainer):
    def __init__(self, input_size, hidden_size=50, num_layers=5, device=None):
        model = GRUModel(input_size, hidden_size, num_layers)
        super().__init__(model, device)
