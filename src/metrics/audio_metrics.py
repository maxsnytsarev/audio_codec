import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

from src.metrics.base_metric import BaseMetric
import torchaudio

class StoiMetric(BaseMetric):
    def __init__(self, sample_rate, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = ShortTimeObjectiveIntelligibility(sample_rate, False)
        self.metric = metric.to(device)

    def __call__(
        self, original_length, data_object: torch.Tensor, logits: torch.Tensor, **kwargs
    ):
        real = data_object.squeeze(1)
        fake = logits.squeeze(1)
        scores = []
        b, _ = real.shape
        for i in range(b):
            L = original_length[i].item()
            scores.append(self.metric(fake[i : i + 1, :L], real[i : i + 1, :L]))
        return torch.stack(scores).mean()


class NISQAMetric(BaseMetric):
    def __init__(self, sample_rate, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = NonIntrusiveSpeechQualityAssessment(sample_rate)
        self.metric = metric.to(device)

    def __call__(
        self, original_length, data_object: torch.Tensor, logits: torch.Tensor, **kwargs
    ):
        fake = logits.squeeze(1)
        b, _ = fake.shape
        scores = []
        for i in range(b):
            L = original_length[i].item()
            scores.append(self.metric(fake[i : i + 1, :L]))
        return torch.stack(scores).mean()


class LogMELMetric(BaseMetric):
    def __init__(self, sample_rate, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                win_length=1024,
                hop_length=1024 // 4,
                n_mels=64,
                power=1.0,
            )
        self.mel_spec = self.mel_spec.to(device)

    def __call__(
        self, original_length, data_object: torch.Tensor, logits: torch.Tensor, **kwargs
    ):
        real = data_object.squeeze(1)
        fake = logits.squeeze(1)
        b, _ = fake.shape
        scores = []
        for i in range(b):
            L = original_length[i].item()
            real_spec = self.mel_spec(real[i : i + 1, :L])
            fake_spec = self.mel_spec(fake[i : i + 1, :L])
            eps = 1e-12
            log_res = torch.log(real_spec + eps) - torch.log(fake_spec + eps)
            scores.append(log_res.abs().mean())
        return torch.stack(scores).mean()


class LowFreqEnergy(BaseMetric):
    def __init__(self, sample_rate, device, obj_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_freq = 200
        self.obj_type = obj_type
        self.sr = sample_rate

    def energy_tail(self, x):
        x = x - x.mean()
        spec = torch.fft.rfft(x)
        spec = spec.abs() ** 2
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1 / self.sr, device=x.device)
        low_freqs = freqs <= self.max_freq
        all_energy = spec.sum().clamp(min=1e-12)
        low_energy = spec[low_freqs].sum()
        return low_energy / all_energy

    def __call__(
        self, original_length, data_object: torch.Tensor, logits: torch.Tensor, **kwargs
    ):
        real = data_object.squeeze(1)
        fake = logits.squeeze(1)
        b, _ = fake.shape
        scores = []
        for i in range(b):
            L = original_length[i].item()
            if self.obj_type == "real":
                scores.append(self.energy_tail(real[i, :L]))
            elif self.obj_type == "fake":
                scores.append(self.energy_tail(fake[i, :L]))
        return torch.stack(scores).mean()

class HighFreqEnergy(BaseMetric):
    def __init__(self, sample_rate, device, obj_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.min_freq = 4000
        self.obj_type = obj_type
        self.sr = sample_rate

    def energy_tail(self, x):
        x = x - x.mean()
        spec = torch.fft.rfft(x)
        spec = spec.abs() ** 2
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1 / self.sr, device=x.device)
        high_freqs = freqs >= self.min_freq
        all_energy = spec.sum().clamp(min=1e-12)
        high_energy = spec[high_freqs].sum()
        return high_energy / all_energy

    def __call__(
        self, original_length, data_object: torch.Tensor, logits: torch.Tensor, **kwargs
    ):
        real = data_object.squeeze(1)
        fake = logits.squeeze(1)
        b, _ = fake.shape
        scores = []
        for i in range(b):
            L = original_length[i].item()
            if self.obj_type == "real":
                scores.append(self.energy_tail(real[i, :L]))
            elif self.obj_type == "fake":
                scores.append(self.energy_tail(fake[i, :L]))
        return torch.stack(scores).mean()

class SpectralCentroid(BaseMetric):
    def __init__(self, sample_rate, device, obj_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.obj_type = obj_type
        self.sr = sample_rate

    def centroid(self, x):
        x = x - x.mean()
        spec = torch.fft.rfft(x)
        mag = spec.abs()
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1 / self.sr, device=x.device)
        return (freqs * mag).sum() / mag.sum().clamp(min=1e-12)

    def __call__(
        self, original_length, data_object: torch.Tensor, logits: torch.Tensor, **kwargs
    ):
        real = data_object.squeeze(1)
        fake = logits.squeeze(1)
        b, _ = fake.shape
        scores = []
        for i in range(b):
            L = original_length[i].item()
            if self.obj_type == "real":
                scores.append(self.centroid(real[i, :L]))
            elif self.obj_type == "fake":
                scores.append(self.centroid(fake[i, :L]))
        return torch.stack(scores).mean()

class SpectralFlatness(BaseMetric):
    def __init__(self, sample_rate, device, obj_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.obj_type = obj_type
        self.sr = sample_rate

    def flatness(self, x):
        x = x - x.mean()
        spec = torch.fft.rfft(x)
        mag = spec.abs()
        mag = mag.clamp(min=1e-12)
        return torch.exp(torch.log(mag).mean()) / (mag.mean()).clamp(min=1e-12)

    def __call__(
        self, original_length, data_object: torch.Tensor, logits: torch.Tensor, **kwargs
    ):
        real = data_object.squeeze(1)
        fake = logits.squeeze(1)
        b, _ = fake.shape
        scores = []
        for i in range(b):
            L = original_length[i].item()
            if self.obj_type == "real":
                scores.append(self.flatness(real[i, :L]))
            elif self.obj_type == "fake":
                scores.append(self.flatness(fake[i, :L]))
        return torch.stack(scores).mean()