# PruneShift

## Structure

| `jupyter/` | Visualization |
| `pruneshift/` | Provides models, pruning methods and datasets |
| `experiments/` | Lightning scripts for the experiments |

## Workflow
 - Work with vim and jupyter.
 - Jupyter for explorative programming and experiments.
 - rsync to synchronize to dacky.
 - And occasionally push to github.

## Dashboard
A good package for imagenet models can be found here, unfortunately there is no comparable collection for cifar10:
    https://github.com/rwightman/pytorch-image-models

## Current thoughts
- TODOS:
    - Imitate the project structure of click:
        - Move everything into src
        - Create a config file.
        - Make the package structure flat.
    - Log configurations as hyparameters with the lightning module.
    - Implement Oneshot, Rewind, Hydra and one additional state of the art.
    - Add augmix to learn from.


## Zen garden
- We want to do experiments not frameworks!
- Code is also configuration! Thus, click is better than hydra.
- Speed matters!
- Do not mess with gin or hydra :)
