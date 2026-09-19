"""
Assembles CNNFeatureExtractor -> LSTMEncoder -> AdditiveAttention -> Dense
into the full CNN-LSTM-Attention classifier. This is the only file other
code should import the model from (training/, evaluation/, interpretability/
all import CNNLSTMAttention from here, not the sub-modules directly).
"""

import torch.nn as nn

from models.cnn import CNNFeatureExtractor
from models.lstm import LSTMEncoder
from models.attention import AdditiveAttention
from configs import config


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_mels=None, lstm_hidden=128, lstm_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        n_mels = n_mels or config.N_MELS

        self.cnn = CNNFeatureExtractor()

        # 3 poolings of factor 2 -> frequency axis shrinks by 8x
        freq_after_pool = n_mels // 8
        lstm_input_size = 128 * freq_after_pool

        self.lstm = LSTMEncoder(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.attention = AdditiveAttention(self.lstm.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.lstm.output_dim, 1)

    @property
    def last_conv_features(self):
        """Exposed for interpretability/gradcam.py -- the spectral-saliency attribution point."""
        return self.cnn.last_conv_features

    def forward(self, x, return_attention=False):
        """
        x: (batch, 1, n_mels, time_frames) log-mel spectrogram
        Returns logits (batch,); also attn_weights (batch, T') if return_attention=True.
        """
        features = self.cnn(x)  # (batch, 128, F', T')

        batch, C, F, T = features.shape
        seq = features.permute(0, 3, 1, 2).reshape(batch, T, C * F)  # (batch, T, C*F)

        lstm_out = self.lstm(seq)  # (batch, T, lstm_out_dim)
        context, attn_weights = self.attention(lstm_out)  # attn_weights = temporal channel

        out = self.dropout(context)
        logits = self.fc(out).squeeze(-1)  # (batch,)

        if return_attention:
            return logits, attn_weights
        return logits
