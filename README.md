# PruneShift

## Structure

| `jupyter/` | Visualization |
| `pruneshift/` | Provides models, pruning methods and datasets |
| `experiments/` | Lightning scripts for the experiments |

## Gin vs Hydra:
Decision we just use the simplest version of click :)
- We want to experiment a lot instead of trying different configuration frameworks.

## Workflow
 - Work with vim and jupyter.
 - Jupyter for explorative programming and experiments.
 - rsync to synchronize to dacky.
 - And occasionally push to github.

## Current thoughts
- TODOS:
    - Make it easier to configure things and reload stuff,
    - Make it possible to load stuff from different angels.
    - Log configurations.
    - Make train runs for different architectures.
    - PROVIDE A BETTER DATAMODULE FOR CIFAR10C
    - Test Magnitude Pruning on the checkpointed networks.

- Optional TODODS:
    - Better config structure:
        Folder structure:
            config/
                main.yaml # Contains basic settings like directory template.

        We can overwrite params.
        https://hydra.cc/docs/patterns/specializing_config
    - Fractor out the code for the experiments into the module code:
        A Trainer Module
    

## Hypothesis
- Current pruning techniques decrease when we have a distr shift:
  - This changes over time, depending on the part we are in training.
  - This effect varies depending on model size and the efficiency (accuracy/size)
   of the model e.g. networks found by NAS might be harder to prune?


## Zen garden
- We want to do experiments not frameworks!
- Code is also configuration!
- Speed matters!
- Do not mess with gin or hydra :)
