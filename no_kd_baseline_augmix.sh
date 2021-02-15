#!/bin/bash


ps_oneshot loss=augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20  prune.method=global_weight prune.ratio=1.1 trainer.max_epochs=30 trainer=ddp

ps_oneshot loss=augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20  prune.method=global_weight prune.ratio=1.3 trainer.max_epochs=30 trainer=ddp

ps_oneshot loss=augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20  prune.method=global_weight prune.ratio=1.5 trainer.max_epochs=30 trainer=ddp

ps_oneshot loss=augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20  prune.method=global_weight prune.ratio=2 trainer.max_epochs=30 trainer=ddp

ps_oneshot loss=augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20  prune.method=global_weight prune.ratio=4 trainer.max_epochs=30 trainer=ddp

ps_oneshot loss=augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20  prune.method=global_weight prune.ratio=8 trainer.max_epochs=30 trainer=ddp

ps_oneshot loss=augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20  prune.method=global_weight prune.ratio=16 trainer.max_epochs=30 trainer=ddp



