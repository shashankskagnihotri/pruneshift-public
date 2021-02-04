import logging

import hydra
from hydra.utils import instantiate
from hydra.utils import call

from pruneshift.scripts.utils import create_trainer
from pruneshift.scripts.utils import save_config
from pruneshift.scripts.utils import partial_instantiate
from pruneshift.datamodules import datamodule
from pruneshift.modules import VisionModule


@hydra.main(config_path="configs", config_name="train.yaml")
def train(cfg):
    """ Trains neural networks."""
    save_config(cfg)
    trainer = create_trainer(cfg)
    # Currently we should initialize the trainer first because of the seed.
    network = call(cfg.network)
    data = datamodule(**cfg.datamodule)
    optimizer_fn = partial_instantiate(cfg.optimizer)
    scheduler_fn = partial_instantiate(cfg.scheduler)
    train_loss = instantiate(cfg.loss)

    module = VisionModule(network=network,
                          test_labels=data.labels,
                          optimizer_fn=optimizer_fn,
                          scheduler_fn=scheduler_fn,
                          train_loss=train_loss)

    trainer.fit(module, datamodule=data)
    trainer.test(module, datamodule=data)


if __name__ == "__main__":
    train()

