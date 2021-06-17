import logging
from functools import partial

import hydra
from hydra.utils import instantiate
from hydra.utils import call

from pruneshift.scripts.utils import create_trainer
from pruneshift.scripts.utils import save_config
from pruneshift.scripts.utils import partial_instantiate
from pruneshift.scripts.utils import create_loss
from pruneshift.scripts.utils import print_test_results
from pruneshift.modules import VisionModule 
from pruneshift.datamodules import ShiftDataModule
from pruneshift.prune_hydra import hydrate


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="subnet.yaml")
def subnet(cfg):
    """ Prunes a network and than finetunes it."""
    save_config(cfg)
    # Note that the trainer must be created first, for the seeding.
    trainer = create_trainer(cfg)
    network = call(cfg.network)
    data = ShiftDataModule(**cfg.datamodule)
    optimizer_fn = partial_instantiate(cfg.optimizer)
    scheduler_fn = partial_instantiate(cfg.scheduler)
    train_loss = create_loss(cfg, network, data)

    logger.info("Bring the module into hydra state!")
    hydrate(network, **cfg.subnet)

    module = VisionModule(
        network=network,
        datamodule=data,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        train_loss=train_loss,
    )

    trainer.fit(module, datamodule=data)
    results = trainer.test(module, datamodule=data, verbose=False)
    print_test_results(results)


if __name__ == "__main__":
    subnet()
