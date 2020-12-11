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

## Current thoughts
- TODOS:
    - Make a training run where we checkpoint each iteration.
    - We use ResNet50

    

## Hypothesis
- Current pruning techniques decrease when we have a distr shift:
  - This changes over time, depending on the part we are in training.
  - This effect varies depending on model size and the efficiency (accuracy/size)
   of the model e.g. networks found by NAS might be harder to prune?


## Zen garden
- We want to do experiments not frameworks!
- Code is also configuration!
- Speed matters!
