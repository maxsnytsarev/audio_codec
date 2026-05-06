import torch.nn as nn
import torch.nn.functional as F


class casualConv(nn.Module):
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

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
