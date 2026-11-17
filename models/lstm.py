"""
Bidirectional LSTM over the CNN's reshaped output sequence -- models how
spectral patterns evolve across the clip. Chosen over a vanilla RNN because
vanilla RNNs lose track of anything more than a few dozen steps back
(vanishing gradients); LSTM's gating fixes that.
"""

import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        """x: (batch, T, input_size) -> (batch, T, output_dim)"""
        out, _ = self.lstm(x)
        return out
