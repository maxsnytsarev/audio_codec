import torch
from torch import nn

from src.transforms.crop import Crop
from src.transforms.pad import Pad


class Transform(nn.Module):
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
        self.crop = Crop()
        self.pad = Pad()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        return self.pad(self.crop(x))
