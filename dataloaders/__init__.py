from dataloaders.experiments import *
from configs import Config


def get_dataloader(cfg: Config):
    return globals().get(cfg.experiment.name)