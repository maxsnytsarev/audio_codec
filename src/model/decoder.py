import torch.nn as nn
from src.model.encoder_model import ResidualUnit
from src.model.CausalConvolution import causalConv, causalConvTranspose

class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()
        self.conv = causalConvTranspose(in_channels=out_channels * 2, out_channels=out_channels, kernel=2 * stride, stride=stride)
        self.resunit1 = ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1)
        self.resunit2 = ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3)
        self.resunit3 = ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9)
    def forward(self, input):
        x = self.conv(input)
        x = self.resunit1(x)
        x = self.resunit2(x)
        x = self.resunit3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.conv1 = causalConv(in_channels=D, out_channels=16 * C, kernel=7, stride=1, dilation=1)
        self.blocks = nn.ModuleList()
        self.blocks.append(DecoderBlock(out_channels=8 * C, stride=5))
        self.blocks.append(DecoderBlock(out_channels=4 * C, stride=5))
        self.blocks.append(DecoderBlock(out_channels=2 * C, stride=4))
        self.blocks.append(DecoderBlock(out_channels=1 * C, stride=2))
        self.conv2 = causalConv(in_channels=C, out_channels=1, kernel=7, stride=1, dilation=1)

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
