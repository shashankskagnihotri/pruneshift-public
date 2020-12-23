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
    - Reintroduce gin-config:
        - Most of the time we try to write factory code, and parsing the correct
            configurations is very very messy:
            eg we wanted to specifically configure some setting and its not that great.
        - This will also allow us to have a huge flexibility regarding creating datamodules.
        - Hydra is too fancy for us.
    - Finish folder structure:
        - Save everything in a specific folder.
    - Implement Oneshot, Rewind, Hydra and one additional state of the art.
        - Oneshot:
            pass
    - Add augmix to learn from.


## Zen garden
- We want to do experiments not frameworks!
- Code is also configuration! Thus, click is better than hydra.
- Speed matters!
- Do not mess with gin or hydra :) But hydra was better in the end!
