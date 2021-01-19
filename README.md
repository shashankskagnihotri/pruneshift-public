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

We leave our gradient based methods as they are harder to work with.

Learning Rate Warmup can be easily done by LambdaLR scheduler.

## TODOS:
- Update the correct learning shedule [Saturday]:
     - Integrate the correct optimizer, only nesterov sgd with momentum. 20 min
     - Add augmix loss 20 min
     - This should be done with hierachical configurations 2h
- Look into the hydra code [Sunday]
- Read the new pruning paper.


## Questions
1. Is ood generalization degrading more when using pruning? Missing method to read data out.
2. Is there any big difference in the simple base methods?
3. How does ood generalization evolves over time?
4. And how does it work with pruning? When we overfit what happens?
5. Does finetuning with augmix help?
6. Does finetuning with augmix combined with hydra help?


## Zen Garden
Drink Tea.

