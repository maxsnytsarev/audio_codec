import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import GanTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="soundstream")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)
    generator_loss = instantiate(config.generator_criterion).to(device)
    discriminator_loss = instantiate(config.discriminator_criterion).to(device)
    # get function handles of loss and metrics
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    generator_optimizer = instantiate(
        config.generator_optimizer, params=model.generator.parameters()
    )
    discriminator_optimizer = instantiate(
        config.discriminator_optimizer,
        params=list(model.wave_discriminator.parameters())
        + list(model.stft_discriminator.parameters()),
    )
    generator_lr_scheduler = instantiate(
        config.generator_lr_scheduler, optimizer=generator_optimizer
    )
    discriminator_lr_scheduler = instantiate(
        config.discriminator_lr_scheduler, optimizer=discriminator_optimizer
    )

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")
    trainer = GanTrainer(
        model=model,
        generator_criterion=generator_loss,
        discriminator_criterion=discriminator_loss,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_lr_scheduler=generator_lr_scheduler,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
        metrics=metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
