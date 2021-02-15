#!/bin/bash

ps_train loss=augmix datamodule=cifar100_augmix network.network_id=cifar100_resnet56 trainer=ddp path=augmix