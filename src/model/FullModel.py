import torch.nn as nn

from src.model.Discriminator import STFTDiscriminator, WaveDiscriminator
from src.model.Generator import Generator


class FullModel(nn.Module):
    def __init__(self, C, D, N_q):
        super().__init__()
        self.C = C
        self.D = D
        self.N_q = N_q
        self.generator = Generator(C=C, D=D, N_q=N_q)
        self.wave_discriminator = WaveDiscriminator()
        self.stft_discriminator = STFTDiscriminator()

    def generate(self, data_object, **batch):
        return self.generator(data_object)

    def discriminate(self, data_object, logits, **batch):
        real = data_object
        fake = logits
        result = {}
        wave_real = self.wave_discriminator(real)
        stft_real = self.stft_discriminator(real)
        result["disc_real"] = wave_real["logits"]
        result["feat_d_real"] = wave_real["feature_maps"]
        result["stft_real"] = stft_real["logits"]
        result["feat_stft_real"] = stft_real["feature_maps"]
        wave_fake = self.wave_discriminator(fake)
        stft_fake = self.stft_discriminator(fake)
        result["disc_false"] = wave_fake["logits"]
        result["feat_d_false"] = wave_fake["feature_maps"]
        result["stft_false"] = stft_fake["logits"]
        result["feat_stft_false"] = stft_fake["feature_maps"]
        return result

    def forward(self, data_object, **batch):
        return self.generator(data_object)

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
