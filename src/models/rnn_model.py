import torch.nn as nn
from .base_trainer import BaseTrainer

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=5):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

class RNNTrainer(BaseTrainer):
    def __init__(self, input_size, hidden_size=50, num_layers=5, device=None):
        model = RNNModel(input_size, hidden_size, num_layers)
        super().__init__(model, device)
