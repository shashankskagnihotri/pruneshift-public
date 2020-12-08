"""Our first basic experiment."""
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl

from pruneshift.models.modules import DistrShift
from pruneshift.models.topology import conv_n_lottery
from pruneshift.data.datamodule import CIFAR10DistrShift


@hydra.main(config_name="config")
def experiment(cfg):
    model = DistrShift(conv_n_lottery(2))
    data = CIFAR10DistrShift(cfg.DataModule.dataset_path, "fog", lvl=3)
    trainer: pl.Trainer = instantiate(cfg.Trainer)
    # TODO: Split the datamodule such that we make different test calls :)
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
    trainer.test()


if __name__ == "__main__":
    experiment()

