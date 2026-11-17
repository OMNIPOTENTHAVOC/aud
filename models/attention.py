"""
Bahdanau-style additive attention over the LSTM's output sequence.
Learns which time steps matter most, and doubles as the temporal-saliency
channel for interpretability -- no separate mechanism needed to answer
"which moments did the model lean on."
"""

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        """lstm_out: (batch, T, hidden_dim) -> context: (batch, hidden_dim), weights: (batch, T)"""
        energy = torch.tanh(self.W(lstm_out))
        scores = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights
