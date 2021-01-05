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
    - Look into shrinkbench.
        - We should not prune the readout layer.
        - They use the original resnet variant from the paper, which is a variant without batchnorm.
        - They calculate the pruning ratio regarding all learnable parameters: eg They take into account the batchnorm layers and readout layers.
        - How can we integrate this into our own code base?
        - They also prune only conv and linear layers.
    - Implement evaluate. 2h
    - Implement rewind. 2h
    - Integrate structured pruning. 2h
        - Here we should read a little bit about structured pruning.
    - Integrate augmix. 1h
    - Visualize where weights are pruned.

## Next Meeting:
    - Pruning or strutctured pruning?

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
