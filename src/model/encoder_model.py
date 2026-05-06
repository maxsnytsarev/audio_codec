import torch.nn as nn
from CausalConvolution import causalConv


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = causalConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel=7,
            stride=1,
            dilation=dilation,
        )
        self.conv2 = causalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=1,
            stride=1,
            dilation=1,
        )
        self.activation = nn.ELU()

    def forward(self, x):
        y = self.conv2(self.activation(self.conv1(x)))
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.resunit1 = ResidualUnit(in_channels // 2, in_channels // 2, dilation=1)
        self.resunit2 = ResidualUnit(in_channels // 2, in_channels // 2, dilation=3)
        self.resunit3 = ResidualUnit(in_channels // 2, in_channels // 2, dilation=9)
        self.conv = causalConv(
            in_channels // 2,
            out_channels=in_channels,
            kernel=2 * stride,
            stride=stride,
            dilation=1,
        )

    def forward(self, x):
        x = self.resunit1(x)
        x = self.resunit2(x)
        x = self.resunit3(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.conv1 = causalConv(
            in_channels=1, out_channels=C, kernel=7, stride=1, dilation=1
        )
        self.blocks = nn.ModuleList()
        self.blocks.append(EncoderBlock(2 * C, stride=2))
        self.blocks.append(EncoderBlock(4 * C, stride=4))
        self.blocks.append(EncoderBlock(8 * C, stride=5))
        self.blocks.append(EncoderBlock(16 * C, stride=5))
        self.conv2 = causalConv(
            in_channels=16 * C, out_channels=D, kernel=3, stride=1, dilation=1
        )

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        x = self.conv1(data_object)
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)
        return {"logits": x}

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
