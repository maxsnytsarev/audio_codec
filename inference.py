import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_codec")
def main(config):
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    # get function handles of loss and metrics
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler

    save_path = config.inferencer.get("save_path")
    save_path = Path(save_path)
    inferencer = Inferencer(
        model=model,
        metrics=metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
    )

    inferencer.run_inference()


if __name__ == "__main__":
    main()
