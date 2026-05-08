import math

import torch
import torchaudio
from torch import nn


class GeneratorLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr
        self.log_s = list(range(6, 12))
        self.mel_specs = nn.ModuleList()
        for log_s in self.log_s:
            s = 2**log_s
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=s,
                win_length=s,
                hop_length=s // 4,
                n_mels=64,
                power=1.0,
            )
            self.mel_specs.append(mel_spec)

    def forward(
        self,
        data_object,
        logits,
        stft_real,
        stft_false,
        feat_stft_real,
        feat_stft_false,
        disc_real,
        disc_false,
        feat_d_real,
        feat_d_false,
        commitment_loss,
        **batch
    ):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        real = data_object.squeeze(1)
        fake = logits.squeeze(1)
        l_rec = torch.tensor([0.0]).to(logits.device)
        i = 0
        for log_s in self.log_s:
            s = 2**log_s
            alpha = math.sqrt(s // 2)
            mel_spec = self.mel_specs[i]
            real_spec = mel_spec(real)
            fake_spec = mel_spec(fake)
            res = real_spec - fake_spec
            eps = 1e-12
            log_res = torch.log(real_spec + eps) - torch.log(fake_spec + eps)
            l_rec += (
                torch.linalg.norm(res, dim=1, ord=1).mean()
                + alpha * torch.linalg.norm(log_res, dim=1, ord=2).mean()
            )
            i += 1

        l_adv = torch.tensor([0.0]).to(logits.device)
        false_logits = [stft_false] + disc_false
        for k in range(len(false_logits)):
            cur = torch.relu(1 - false_logits[k]).flatten(start_dim=1)
            T_k = cur.shape[-1]
            l_adv += (1 / T_k) * cur.sum(-1).mean()
        l_adv /= len(false_logits)
        L = 6
        l_feat = torch.tensor([0.0]).to(logits.device)
        real_fm = feat_stft_real + feat_d_real
        false_fm = feat_stft_false + feat_d_false
        for k in range(len(false_logits)):
            for l_ in range(L):
                cur = torch.abs(real_fm[k][l_].detach() - false_fm[k][l_]).flatten(
                    start_dim=1
                )
                T = cur.shape[-1]
                l_feat += (1 / T) * cur.sum(dim=-1).mean()
        l_feat /= len(false_logits) * L
        loss = 1 * l_adv + 100 * l_feat + 1 * l_rec + 1 * commitment_loss
        return {
            "loss": loss,
            "generator_loss": loss.detach(),
            "adv_loss": l_adv.detach(),
            "feat_loss": l_feat.detach(),
            "rec_loss": l_rec.detach(),
            "commitment_loss": commitment_loss.detach(),
        }


class DiscriminatorLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, stft_real, stft_false, disc_real, disc_false, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        loss = torch.tensor([0.0]).to(logits.device)
        false_logits = [stft_false] + disc_false
        real_logits = [stft_real] + disc_real
        for k in range(len(false_logits)):
            cur1 = torch.relu(1 - real_logits[k]).flatten(start_dim=1)
            cur2 = torch.relu(1 + false_logits[k]).flatten(start_dim=1)
            cur = cur1 + cur2
            T_k = cur.shape[-1]
            loss += (1 / T_k) * cur.sum(-1).mean()
        loss /= len(false_logits)
        return {"loss": loss, "discriminator_loss": loss.detach()}
