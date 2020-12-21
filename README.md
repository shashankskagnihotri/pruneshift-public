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
        - MOVE EVERYTHING INTO SRC
        - CREATE A CONFIG FILE.
        - MAKE THE CLI TO ONE FILE ONLY AND MOVE IT INTO PRUNESHIFT.
        - Export the oneshot function call and stuff like this into a run file.
        - MAKE THE PACKAGE STRUCTURE FLAT.
        - INTRODUCE ANOTHER SRC FOR CIFAR10_MODELS.
        - FIND OUT WHEN THE LIGHTNING LOGGER IS INITIALIZED.
        - LOG HYPARAMETERS WITH THE LIGHTNING MODULE.
        - Check whether the CSV logger does work right.
        - Introduce subfolder structure depending on the command.
        - Set experiment names.
        
    - Implement Oneshot, Rewind, Hydra and one additional state of the art.
        - Oneshot:
            pass
    - Add augmix to learn from.


## Zen garden
- We want to do experiments not frameworks!
- Code is also configuration! Thus, click is better than hydra.
- Speed matters!
- Do not mess with gin or hydra :)
