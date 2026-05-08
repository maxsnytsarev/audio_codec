import torch
from torchmetrics.audio import (
    NonIntrusiveSpeechQualityAssessment,
    ShortTimeObjectiveIntelligibility,
)

from src.metrics.base_metric import BaseMetric


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
