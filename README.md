# PruneShift

## Structure

| `jupyter/` | Visualization |
| `pruneshift/` | Provides models, pruning methods and datasets |
| `experiments/` | Lightning scripts for the experiments |

## Workflow
 - Work with vim and jupyter.
 - Jupyter for explorative programming and experiments.
 - rsync to synchronize to dacky.
 - And occasionally git to github.

## Current thoughts
- TODOS:
    - Reading the ImageNet-C paper completely.
    - DATAMODULE IS MISSING ~ 1H
    - LIGHTNINGMODULE IS MISSING ~ 1H
    - 1. MNIST pruning:
        - Training sheme is already done.
        - After we finished one experiment, we have to reload it.
        - Just make this explicit by requiring a folder where everything is saved.
    - 1. WRITE A BASIC PRUNING MODULE.
    - 2. Make a basic run and than prune percentage wise.
    - Pruning methods that we want to integrate ~ 3h:
        SyncFlow, Structured Pruning and so on..
    - Look into lsyncd.

## Zen garden
- We want to do experiments not frameworks!
- Code is also configuration!
- Speed matters!
