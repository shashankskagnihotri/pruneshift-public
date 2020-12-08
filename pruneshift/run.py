import os
from typing import Dict

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pruneshift.models import MNISTModel
from pruneshift.data import MNISTDataModule


# TODO: This method should work now, but should be tested,
#       whether the forward prediction is correct.
#       Probably, the pruning is only correct after the first forward pass.
#       What about nested modules?

def lottery_ticket(model: nn.Module,
                   init_state_dict: Dict,
                   pruning_mode: str,
                   amount: float) -> nn.Module:
    """Creates a lottery ticket (inplace).

    Args:
        model: Model to prune.
        init_state_dict: The initial state of the model.
        pruning_mode: Which pruning_fn to use.
        amount: The amount of pruning.

    Returns:
        The pruned version
    """
    pairs = [(s, n) for s in model.children() for n, _ in s.named_parameters()]
    prune.global_unstructured(pairs, prune.L1Unstructured, amount=amount)
    # Load the inital params.
    model.load_state_dict({n + "_orig": p for n, p in init_state_dict.items()},
                          strict=False)
    return model


@hydra.main(config_name="config")
def main(cfg):
    model = MNISTModel()
    data: MNISTDataModule = instantiate(cfg.DataModule)

    model.named_parameters()

    filename = "{epoch:02d}-{val_loss:.2f}"
    chpt_callback = ModelCheckpoint(monitor="Validation/Loss",
                                    save_top_k=3,
                                    filename=filename,
                                    mode="min")

    trainer: pl.Trainer = instantiate(cfg.Trainer, callbacks=[chpt_callback])
    # Save initiliazation.
    chpt_path = os.path.join(trainer.default_root_dir, "checkpoint")
    torch.save(model.state_dict(), chpt_path)
    # Train the network.
    trainer.fit(model, datamodule=data)
    # Load the best performing model.
    path = chpt_callback.best_model_path
    model = MNISTModel.load_from_checkpoint(path)
    # Prune the model.
    lottery_ticket(model, torch.load(chpt_path), "Blub", 0.3)


if __name__ == "__main__":
    main()
