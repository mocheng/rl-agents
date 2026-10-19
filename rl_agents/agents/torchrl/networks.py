import numpy as np
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """
        An SB3-CnnPolicy-style convolutional network for channel-first image observations:
        3 conv layers (32@8x8/4, 64@4x4/2, 64@3x3/1) followed by a 512-unit fully-connected
        layer and the action head. Observations are cast to float and scaled to [0, 1].

        This mirrors the capacity and initialization of the network used by the repository's
        working CarRacing DQN (Stable-Baselines3 MSEDQN), which is ~100x larger than the
        repository's own `ConvolutionalNetwork`.
    """

    def __init__(self, in_channels, out_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, in_channels, 84, 84)).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512), nn.ReLU(),
            nn.Linear(512, out_features),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = x.float() / 255.0
        if x.dim() == 3:  # unbatched (C, H, W) observation from a single environment
            x = x.unsqueeze(0)
        return self.fc(self.cnn(x))
