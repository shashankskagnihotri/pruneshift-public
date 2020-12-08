Start an experiment with
```bash
python -m pruneshift.run
```

Accessing System Variables can be done by the following:
```python
os.environ["VarName"]
```
Framework to prune. We use the pytorch pruner

Data Module can return multiple loaders for different test sets the `test_step` argument
than is able to find out the idx of the `data_loader`.
    

*Workflow*
 - Work with vim and jupyter.
 - Jupyter for explorative programming and experiments.
 - rsync to synchronize to dacky.
 - And occasionally git to github.

*Current thoughts*
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

*Open questions*
1. Differences between structural pruning and NAS.

*Zen garden*
- flat structures!
- 

