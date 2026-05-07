import torch.nn as nn
import torch


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, stride=1, kernel_size=15)
        self.lrelu = nn.LeakyReLU(0.2)
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Conv1d(in_channels=16, out_channels=64, stride=4, kernel_size=41, groups=4))
        self.blocks.append(nn.Conv1d(in_channels=64, out_channels=256, stride=4, kernel_size=41, groups=16))
        self.blocks.append(nn.Conv1d(in_channels=256, out_channels=1024, stride=4, kernel_size=41, groups=64))
        self.blocks.append(nn.Conv1d(in_channels=1024, out_channels=1024, stride=4, kernel_size=41, groups=256))
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=1024, stride=1, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=1, stride=1, kernel_size=3)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        for block in self.blocks:
            x = self.lrelu(block(x))
        x = self.lrelu(self.conv2(x))
        x = self.conv3(x)
        return x


class WaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disk1 = DiscriminatorBlock(in_channels=1)
        self.disk2 = DiscriminatorBlock(in_channels=1)
        self.disk3 = DiscriminatorBlock(in_channels=1)
        self.pool= nn.AvgPool1d(kernel_size=4)
    def forward(self, x):
        x1 = self.disk1(x)
        x2 = self.disk2(self.pool(x))
        x3 = self.disk3(self.pool(self.pool(x)))
        return {"logits": [x1, x2, x3]}


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


class ResidualUnitDick(nn.Module):
    def __init__(self, in_channels, m, s):
        super().__init__()
        self.in_channels = in_channels
        self.m = m
        st, sf = s
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=m * in_channels, kernel_size=(st + 2, sf + 2),stride=(st, sf))
        self.lrelu = nn.LeakyReLU(0.2)
        self.skip = nn.Conv2d( in_channels=in_channels, out_channels=m * in_channels, kernel_size=(st + 2, sf + 2), stride=(st, sf))
    def forward(self, x):
        y = x.clone()
        x = self.lrelu(self.conv1(x))
        x = self.conv2(x)
        return self.skip(y) + x

class STFTDiscriminator(nn.Module):
    def __init__(self, w, h, F):
        super().__init__()
        self.F = F
        self.register_buffer("window", torch.hann_window(w))
        self.w = w
        self.h = h
        C = 2
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualUnitDick(in_channels=C, m=2, s=(1, 2)))
        self.blocks.append(ResidualUnitDick(in_channels=2 * C, m=2, s=(2, 2)))
        self.blocks.append(ResidualUnitDick(in_channels=4 * C, m=1, s=(1, 2)))
        self.blocks.append(ResidualUnitDick(in_channels=4 * C, m=2, s=(2, 2)))
        self.blocks.append(ResidualUnitDick(in_channels=8 * C, m=1, s=(1, 2)))
        self.blocks.append(ResidualUnitDick(in_channels=8 * C, m=2, s=(2, 2)))
        self.conv2 = nn.Conv2d(in_channels=16 * C, out_channels=1, kernel_size=(1, F // 2 ** 6))
    def forward(self, x):
        x = x.squeeze(1)
        x = torch.stft(x, self.w, hop_length=self.h, window=self.window, return_complex=True)
        x = torch.stack([x.real, x.imag], dim=1)
        x = x.permute(0, 1, 3, 2)
        for block in self.blocks:
            x = block(x)
        return {"logits": self.conv2(x)}


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
