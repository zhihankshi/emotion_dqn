"""
Small DQN network for low-resolution (e.g. 7x7) pixel observations.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class SmallDQNNetwork(nn.Module):
    """
    Lightweight CNN for small image inputs (e.g. 7x7 with 1 pixel per cell).

    Architecture:
        Conv1: 3 -> 16, 3x3, stride=1, padding=1  (preserves spatial dims)
        Conv2: 16 -> 32, 3x3, stride=1, padding=1  (preserves spatial dims)
        FC1: flattened -> 128
        FC2: 128 -> n_actions
    """

    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super().__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        h, w, c = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_output_size(c, h, w)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def _get_conv_output_size(self, c: int, h: int, w: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            return int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
