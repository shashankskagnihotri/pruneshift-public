# GitHub repo used for [Towards Improving Robustness of Compressed CNNs](https://lmb.informatik.uni-freiburg.de/Publications/2021/SB21/), ICML UDL, 2021 by [Jasper Hoffmann](https://nr.uni-freiburg.de/people/jasper-hoffmann)\*, [Shashank Agnihotri](https://www.uni-mannheim.de/dws/people/researchers/phd-students/shashank/)\*, [Tonmoy Saikia](https://scholar.google.de/citations?user=HHv75fUAAAAJ&hl=en), [Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox/).

## This code is the public fork from [JasperHoffmann/pruneshift](https://github.com/JasperHoffmann/pruneshift) with some changes. 

## [Jasper Hoffmann](https://nr.uni-freiburg.de/people/jasper-hoffmann) was the one majorly responsible for development of this codebase with some modifications from me.

# How to use the repository

1. Steps to integrate Knowledge distillation.
    1. Define a new loss module in src/pruneshift/losses.py
    2. Add a configuration .yaml file to src/pruneshift/scripts/loss:
        _target_: pruneshift.losses.YourLossClass
        some: arguments
    3. If you have done this we can use it eveywhere, for hydra and so on.



To do oneshot pruning with a normal run on cifar100 with a corrupted testset do:
Standard settings are `global_weight` with a ratio of `2`:
```
ps_oneshot prune.method=l1_channels prune.ratio=16
```
To do oneshot pruning with a specific checkpointed network with cifar100 augmix and corrupted test set:
```
ps_oneshot datamodule=cifar100_augmix network.ckpt_path=path/to/somewhere
```
Note, there can not be any hypens in the path to the checkpoint. 

To determine the logdir:
```
ps_oneshot path.logdir=path/to/somewhere
```
Start with multiple gpus:
```
ps_oneshot trainer=ddp 
```

