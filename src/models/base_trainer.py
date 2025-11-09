import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class BaseTrainer:
    """Classe base para treino e predição PyTorch"""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, X_train, y_train, epochs=50, batch_size=32, lr=0.001):
        """Treina o modelo PyTorch"""
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)

        # Garante que sempre retorna shape (num_samples,)
        return outputs.cpu().numpy().reshape(-1)