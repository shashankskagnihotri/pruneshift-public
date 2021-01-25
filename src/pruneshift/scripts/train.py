import logging

import hydra
from hydra.utils import instantiate
from hydra.utils import call

from pruneshift.scripts.utils import create_trainer
from pruneshift.scripts.utils import save_config
from pruneshift.scripts.utils import create_optim

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train.yaml")
def train(cfg):
    """ Trains neural networks."""
    save_config(cfg)
    trainer = create_trainer(cfg)
    # Currently we should initialize the trainer first because of the seed.
    network = call(cfg.network)
    data = call(cfg.datamodule)
    optim_args = create_optim(cfg)
    module = instantiate(cfg.module, network, data.labels, **optim_args)

    logger.info("Starting with training...")
    trainer.fit(module, datamodule=data)
    logger.info("Starting with testing...")
    trainer.test(module, datamodule=data)


if __name__ == "__main__":
    train()
