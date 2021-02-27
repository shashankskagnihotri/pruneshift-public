import logging

import hydra
from hydra.utils import instantiate
from hydra.utils import call

from pruneshift.scripts.utils import create_trainer
from pruneshift.scripts.utils import save_config
from pruneshift.scripts.utils import partial_instantiate
from pruneshift.scripts.utils import create_loss
from pruneshift.scripts.utils import print_test_results
from pruneshift.datamodules import ShiftDataModule
from pruneshift.modules import VisionModule


@hydra.main(config_path="configs", config_name="train.yaml")
def train(cfg):
    """ Trains neural networks."""
    save_config(cfg)
    trainer = create_trainer(cfg)
    # Currently we should initialize the trainer first because of the seed.
    network = call(cfg.network, protect_classifier_fn=None)
    data = ShiftDataModule(**cfg.datamodule)
    optimizer_fn = partial_instantiate(cfg.optimizer)
    scheduler_fn = partial_instantiate(cfg.scheduler)
    train_loss = create_loss(cfg, network, datamodule)

    module = VisionModule(network=network,
                          test_labels=data.labels,
                          optimizer_fn=optimizer_fn,
                          scheduler_fn=scheduler_fn,
                          train_loss=train_loss)

    if cfg.trainer.max_epochs > 0:
        trainer.fit(module, datamodule=data)
    test_results = trainer.test(module, datamodule=data, verbose=False)
    print_test_results(test_results)

if __name__ == "__main__":
    train()

