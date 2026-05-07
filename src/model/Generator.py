import torch.nn as nn
import torch
from src.model.encoder_model import Encoder
from src.model.decoder import Decoder
from src.model.RQV import RQV

class Generator(nn.Module):
    def __init__(self, C, D, N_q):
        super().__init__()
        self.encoder = Encoder(C, D)
        self.decoder = Decoder(D, C)
        self.rqv = RQV(N_q, D)

    def forward(self, data_object, **batch):
        x = self.encoder(data_object)["logits"]
        rqv = self.rqv(x)
        x = rqv["logits"]
        x = self.decoder(x)["logits"]
        return {"logits" : x, "commitment_loss": rqv["commitment_loss"], "indeces" : rqv["indeces"]}

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