import logging

import hydra
from hydra.utils import instantiate
from hydra.utils import call

from .utils import create_trainer
from .utils import save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="oneshot.yaml")
def oneshot(cfg):
    """ Prunes one-shot or iteratively."""
    save_config(cfg)
    # Currently we should create the trainer always first.
    trainer = create_trainer(cfg)

    network = call(cfg.network)
    data_train = call(cfg.data.train)
    data_test = call(cfg.data.test)
    if hasattr(data_test, "labels"):
        module = instantiate(cfg.module, network, data_test.labels)
    else:
        module = instantiate(cfg.module, network)
    # print("Building gradients...")
    # batch_gradient(module, data)
   
    if isinstance(cfg.ratios, int):
        ratios = [cfg.ratios]
    else:
        ratios = cfg.ratios

    for ratio in ratios:
        logger.info("Pruning the network.")
        call(cfg.prune, network, ratio=ratio)
        logger.info("Fine-tuning and testing of the network.")
        trainer.fit(module, datamodule=data_train)
        trainer.test(module, datamodule=data_test)


if __name__ == "__main__":
    oneshot()

