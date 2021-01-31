import logging
from functools import partial

import hydra
from hydra.utils import instantiate
from hydra.utils import call

from pruneshift.scripts.utils import create_trainer
from pruneshift.scripts.utils import save_config
from pruneshift.scripts.utils import create_optim
from pruneshift.scripts.utils import partial_instantiate
from pruneshift.modules import PrunedModule 


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="oneshot.yaml")
def prune_n_tune(cfg):
    """ Prunes one-shot or iteratively."""
    save_config(cfg)
    # Note that the trainer must be created first.
    trainer = create_trainer(cfg)
    network = call(cfg.network)
    data = call(cfg.datamodule)
    prune_fn = partial_instantiate(cfg.prune)
    optimizer_fn = partial_instantiate(cfg.optimizer)
    scheduler_fn = partial_instantiate(cfg.scheduler)

    module = PrunedModule(network=network,
                          prune_fn=prune_fn,
                          test_labels=data.labels,
                          optimizer_fn=optimizer_fn,
                          scheduler_fn=scheduler_fn,
                          **cfg.module)

    trainer.fit(module, datamodule=data)
    trainer.test(module, datamodule=data)


if __name__ == "__main__":
    prune_n_tune()

