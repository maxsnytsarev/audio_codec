import torch.nn as nn
import torch.nn.functional as F


class causalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dilation):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        x = F.pad(data_object, (self.padding, 0))
        return self.conv(x)


class causalConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = (kernel - 1) * dilation + 1 - stride
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, data_object, **batch):
        x = self.conv(data_object)
        return x[..., :-self.padding]