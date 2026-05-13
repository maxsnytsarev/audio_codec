from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.io_utils import ROOT_PATH
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

class GanTrainer(BaseTrainer):
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        generator_criterion,
        discriminator_criterion,
        generator_optimizer,
        discriminator_optimizer,
        generator_lr_scheduler,
        discriminator_lr_scheduler,
        **kwargs
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        super().__init__(
            model=model,
            criterion=generator_criterion,
            optimizer=generator_optimizer,
            lr_scheduler=generator_lr_scheduler,
            **kwargs
        )

        s = 1024
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=s,
            win_length=s,
            hop_length=s // 4,
            n_mels=64,
            power=1.0,
        ).to(self.device)

        self.generator_criterion = generator_criterion
        self.discriminator_criterion = discriminator_criterion

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            self.model.train()
            metric_funcs = self.metrics["train"]
            generated = self.model.generate(**batch)
            batch.update(generated)

            self.discriminator_optimizer.zero_grad()
            discr = self.model.discriminate(
                data_object=batch["data_object"], logits=batch["logits"].detach()
            )
            batch.update(discr)
            d_loss = self.discriminator_criterion(**batch)
            d_loss["loss"].backward()
            self.discriminator_optimizer.step()
            batch["discriminator_loss"] = d_loss["loss"].detach()

            self.generator_optimizer.zero_grad()
            dicr = self.model.discriminate(
                data_object=batch["data_object"], logits=batch["logits"]
            )
            batch.update(dicr)
            g_loss = self.generator_criterion(**batch)
            g_loss["loss"].backward()
            self.generator_optimizer.step()
            batch.update(g_loss)
            batch["loss"] = g_loss["loss"].detach()
            batch["generator_loss"] = g_loss["loss"].detach()

            if self.discriminator_lr_scheduler is not None:
                self.discriminator_lr_scheduler.step()

            if self.generator_lr_scheduler is not None:
                self.generator_lr_scheduler.step()
        else:
            self.model.eval()
            generated = self.model.generate(**batch)
            batch.update(generated)
            zero = torch.zeros((), device=batch["logits"].device)

            batch["loss"] = zero
            batch["generator_loss"] = zero
            batch["discriminator_loss"] = zero

            batch["adv_loss"] = zero
            batch["feat_loss"] = zero
            batch["rec_loss"] = zero
            batch["commitment_loss"] = zero
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            if not self.is_train and met.name == "NISQA" and self._last_epoch % self.config.trainer["nisqa_every"] != 0:
                continue
            metrics.update(met.name, met(**batch))
        return batch

    def log_perp(self, batch):
        if self.writer is None:
            return
        n_q = 8
        N = 1024
        vals = []
        indeces = batch["indeces"].detach()
        for q in range(n_q):
            flat_indeces = indeces[q].reshape(-1)
            cur_count = torch.bincount(flat_indeces, minlength=N)
            prob = cur_count / cur_count.sum()
            prob = prob[prob > 0]
            vals.append(torch.exp(-(prob * torch.log(prob)).sum()))
        perp = torch.stack(vals).mean() / N
        self.writer.add_scalar("perplexity", perp.item())

    def mel(self, wav):
        with torch.no_grad():
            mel = self.mel_spec(wav)
            mel = torch.log(mel + 1e-12)
            mel = mel.squeeze(0)
            mel = mel.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return image

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        L = int(batch["original_length"][0].item())
        real = batch["data_object"][0]
        fake = batch["logits"][0]
        real = real[:, :L]
        fake = fake[:, :L]
        sr = 16000
        self.writer.add_audio("real_audio", real, sample_rate=sr)
        self.writer.add_audio("generated_audio", fake, sample_rate=sr)
        self.writer.add_image("real_mel", self.mel(real))
        self.writer.add_image("generated_mel", self.mel(fake))