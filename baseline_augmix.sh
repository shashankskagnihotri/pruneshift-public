#!/bin/bash


ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/2021-02-08/22-15-35/checkpoint/epoch_98_val_acc_0.67.ckpt" network.network_id=cifar100_resnet18 trainer=ddp prune.method=l1_channels prune.ratio=1.1

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/2021-02-08/22-15-35/checkpoint/epoch_98_val_acc_0.67.ckpt" network.network_id=cifar100_resnet18 trainer=ddp prune.method=l1_channels prune.ratio=1.3

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/2021-02-08/22-15-35/checkpoint/epoch_98_val_acc_0.67.ckpt" network.network_id=cifar100_resnet18 trainer=ddp prune.method=l1_channels prune.ratio=1.5

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/2021-02-08/22-15-35/checkpoint/epoch_98_val_acc_0.67.ckpt" network.network_id=cifar100_resnet18 trainer=ddp prune.method=l1_channels prune.ratio=2

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/2021-02-08/22-15-35/checkpoint/epoch_98_val_acc_0.67.ckpt" network.network_id=cifar100_resnet18 trainer=ddp prune.method=l1_channels prune.ratio=4

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/2021-02-08/22-15-35/checkpoint/epoch_98_val_acc_0.67.ckpt" network.network_id=cifar100_resnet18 trainer=ddp prune.method=l1_channels prune.ratio=8

ps_oneshot loss=kd_augmix datamodule=cifar100_augmix network.ckpt_path="/misc/lmbraid19/agnihotr/thesis_pruneshift/2021-02-08/22-15-35/checkpoint/epoch_98_val_acc_0.67.ckpt" network.network_id=cifar100_resnet18 trainer=ddp prune.method=l1_channels prune.ratio=16



