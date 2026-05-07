import torch
from torch import nn


class Pad(nn.Module):
    """
    Batch-version of Normalize for 1D Input.
    Used as an example of a batch transform.
    """

    def __init__(self):
        """
        Args:
            mean (float): mean used in the normalization.
            std (float): std used in the normalization.
        """
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        if x.shape[-1] >= 8000:
            return x
        else:
            x = x.repeat(1, 8000 // x.shape[-1] + 1)
            return x[:, :8000]
