import torch
import numpy as np
from .lstm_model import LSTMTrainer
from .gru_model import GRUTrainer
from .rnn_model import RNNTrainer

class Trainer:
    """Treinamento e predição unificado para RF e modelos PyTorch"""

    def __init__(self, model_type='lstm', input_size=None, hidden_size=50, num_layers=5, device=None):
        """
        model_type: 'rf', 'lstm', 'gru', 'rnn'
        input_size: número de features (necessário para PyTorch)
        hidden_size: unidades das camadas recorrentes
        num_layers: número de camadas recorrentes
        device: 'cuda' ou 'cpu', se None escolhe automaticamente
        """
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == 'lstm':
            if input_size is None:
                raise ValueError("input_size obrigatório para LSTM")
            self.trainer = LSTMTrainer(input_size, hidden_size, num_layers, self.device)
        elif model_type == 'gru':
            if input_size is None:
                raise ValueError("input_size obrigatório para GRU")
            self.trainer = GRUTrainer(input_size, hidden_size, num_layers, self.device)
        elif model_type == 'rnn':
            if input_size is None:
                raise ValueError("input_size obrigatório para RNN")
            self.trainer = RNNTrainer(input_size, hidden_size, num_layers, self.device)
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")

    def train_and_predict(self, X_train, y_train, X_test, epochs=50, batch_size=32, lr=0.001):
        """
        Treina o modelo e retorna predições.
        Para PyTorch: X_train e X_test devem ser 3D (samples, timesteps, features)
        """
        self.trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
        return self.trainer.predict(X_test)
