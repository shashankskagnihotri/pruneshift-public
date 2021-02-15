#!/bin/bash

#ps_train loss=augmix datamodule=cifar100_augmix network.network_id=cifar100_resnet56 trainer=ddp path=augmix
ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20 trainer=ddp prune.method=global_weight prune.ratio=1.1 trainer.max_epochs=30

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20 trainer=ddp prune.method=global_weight prune.ratio=1.3 trainer.max_epochs=30

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20 trainer=ddp prune.method=global_weight prune.ratio=1.5 trainer.max_epochs=30

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20 trainer=ddp prune.method=global_weight prune.ratio=2 trainer.max_epochs=30

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20 trainer=ddp prune.method=global_weight prune.ratio=4 trainer.max_epochs=30

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20 trainer=ddp prune.method=global_weight prune.ratio=8 trainer.max_epochs=30

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/checkpoint/epoch_97.ckpt" network.network_id=cifar100_resnet20 trainer=ddp prune.method=global_weight prune.ratio=16 trainer.max_epochs=30



