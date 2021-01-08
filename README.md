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

We leave out gradient based methods as they are hard to work with.

## TODOS:
- Saturday: Is runner day, we implement rewind, evaluate at the end of a run.
- Sunday: Is dataset day, we integrate augmix, Imagenette and ImageNetC.
- Implement rewind. 2h
- Integrate structured pruning. 2h



## Questions
1. Is ood generalization degrading more when using pruning? Missing method to read data out.
2. Is there any big difference in the simple base methods?
3. How does ood generalization evolves over time?
4. And how does it work with pruning? When we overfit what happens?
5. Does finetuning with augmix help?
6. Does finetuning with augmix combined with hydra help?


## Zen garden
- We want to do experiments not frameworks!
- Code is also configuration! Thus, click is better than hydra.
- Speed matters!
- Do not mess with gin or hydra :) But hydra was better in the end!
