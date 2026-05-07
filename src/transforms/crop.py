import random

import torch
from torch import nn


class Crop(nn.Module):
    def __init__(self):
        """
        Args:
            mean (float): mean used in the normalization.
            std (float): std used in the normalization.
        """
        super().__init__()
        self.sr = 16000
        self.crop = 8000

    def forward(self, x):
        if x.shape[-1] <= self.crop:
            return x
        random_start = random.randint(0, x.shape[-1] - self.crop)
        return x[:, random_start : random_start + self.crop]
