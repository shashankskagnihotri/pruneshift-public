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
    """ Prunes one-shot or iteratively."""
    save_config(cfg)
    trainer = create_trainer(cfg)
    # Currently we should initialize the trainer first because of the seed.
    network = call(cfg.network)
    data = call(cfg.datamodule)
    optim_args = create_optim(cfg)
    if hasattr(data, "labels"):
        module = instantiate(cfg.module, network, data.labels, **optim_args)
    else:
        module = instantiate(cfg.module, network, **optim_args)
   
    logger.info("Fine-tuning and testing of the network.")
    trainer.fit(module, datamodule=data)
    trainer.test(module, datamodule=data)


if __name__ == "__main__":
    train()
