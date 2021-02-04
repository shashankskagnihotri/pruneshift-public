1. Steps to integrate Knowledge distillation.
    1. Define a new loss module in src/pruneshift/losses.py
    2. Add a configuration .yaml file to src/pruneshift/scripts/loss:
        _target_: pruneshift.losses.YourLossClass
        some: arguments
    3. If you have done this we can use it eveywhere, for hydra and so on.



To do oneshot pruning with a normal run on cifar100 with a corrupted testset:
```
ps_oneshot
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
