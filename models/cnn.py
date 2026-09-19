"""
CNN feature-extraction module: 3 conv layers, 32->64->128 filters,
BatchNorm -> ReLU -> MaxPool(2x2) after each.

Takes a log-mel spectrogram (batch, 1, n_mels, time_frames) and returns
a feature map (batch, 128, n_mels/8, time_frames/8) -- the CNN is used
standalone here so its output can be handed to the LSTM separately in
model.py, and so this file alone is the Grad-CAM attribution point later.
"""

import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        # Populated on every forward() call -- this is the Grad-CAM
        # attribution point (spectral-saliency channel) used later in
        # interpretability/gradcam.py.
        self.last_conv_features = None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        self.last_conv_features = x  # (batch, 128, F', T')
        return x
