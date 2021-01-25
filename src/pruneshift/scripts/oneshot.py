import logging

import hydra
from hydra.utils import instantiate
from hydra.utils import call

from pruneshift.scripts.utils import create_trainer
from pruneshift.scripts.utils import save_config
from pruneshift.scripts.utils import create_optim

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="oneshot.yaml")
def oneshot(cfg):
    """ Prunes one-shot or iteratively."""
    save_config(cfg)
    # Currently we should create the trainer always first.
    trainer = create_trainer(cfg)
    network = call(cfg.network)
    data = call(cfg.datamodule)
    optim_args = create_optim(cfg)

    module = instantiate(cfg.module, network, data.labels, **optim_args)

    ratios = cfg.prune.ratio

    if isinstance(ratios, (int, float)):
        ratios = [ratios]

    for ratio in ratios:
        logger.info("Starting with pruning...")
        info = call(cfg.prune, module, ratio=ratio)
        logger.info("\n %r" % info.summary())
        # module = instantiate(cfg.module, network, data.labels, **optim_args)

        logger.info("Fine-tuning of the network...")
        trainer.fit(module, datamodule=data)

    logger.info("Starting with testing...")
    trainer.test(module, datamodule=data)


if __name__ == "__main__":
    oneshot()
