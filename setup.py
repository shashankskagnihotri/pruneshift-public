from setuptools import setup, find_packages

setup(
    name="PruneShift",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
       "pandas", "torchvision", "torch", "pytorch_lightning", "click" 
    ],
    entry_points='''
        [console_scripts]
        pruneshift=experiments.cli:cli
    ''',
)
