import torch.nn as nn
from .base_trainer import BaseTrainer

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1)

class LSTMTrainer(BaseTrainer):
    def __init__(self, input_size, hidden_size=50, num_layers=5, device=None):
        model = LSTMModel(input_size, hidden_size, num_layers)
        super().__init__(model, device)
