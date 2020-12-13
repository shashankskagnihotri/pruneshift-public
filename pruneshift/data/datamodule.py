""" Provides the data modules we need for our experiments."""
import abc
from typing import Sequence

from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.datasets as torch_datasets
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import CIFAR10C


def datamodule(
    name: str, root: str, batch_size: int = 32, num_workers: int = 5, **kwargs
) -> pl.LightningDataModule:
    """Creates a LightningDataModule.

    Args:
        name: Name of the dataset/datamodule.
        root: Where to download the data if necessary.
        batch_size: The batch size used for the datamodule.
        num_workers: Number of workers used for the dataloaders.
    Returns:
        The corresponding LightningDataModule.
    """
    return BaseDataModule.subclasses[name](root, batch_size, num_workers, **kwargs)


class BaseDataModule(pl.LightningDataModule):
    subclasses = {}

    def __init__(self, root: str, batch_size: int, num_workers: int):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_dataloader(self, dataset):
        if isinstance(dataset, (tuple, list)):
            # If not a dataset we assume a list/tuple of datasets.
            return [self._create_dataloader(d) for d in dataset]
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def __init_subclass__(cls, **kwargs):
        # Add subclasses to the module.
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.name] = cls

    def transform(self, train: bool = False):
        return transforms.ToTensor()

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)


class CIFAR10Module(BaseDataModule):
    name = "CIFAR10"

    def prepare_data(self):
        torch_datasets.CIFAR10(self.root, train=True, download=True)
        torch_datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self, stage):
        create_fn = torch_datasets.CIFAR10
        self.test_dataset = create_fn(self.root, transform=self.transform())

        trainset = create_fn(self.root, transform=self.transform(True))
        trainset, valset = random_split(trainset, [45000, 5000])
        self.train_dataset = trainset
        self.val_dataset = valset

    def transform(self, train: bool = False):
        transform = []
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)

        if train:
            transform.append(transforms.RandomCrop(size=32, padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))

        return transforms.Compose(transform)


class CIFAR10CModule(CIFAR10Module):
    name = "CIFAR10Corrupted"

    @property
    def labels(self):
        """Returns labels of the datasets currently in use."""
        labels = ["undistorted"]
        for distortion in CIFAR10C.distortions_list:
            for lvl in range(1, 6):
                labels.append("{}.{}".format(distortion, lvl))
        return labels

    def prepare_data(self):
        # It is enough to download one
        torch_datasets.CIFAR10(self.root, train=False, download=True)
        CIFAR10C(self.root, "snow")

    def setup(self, stage):
        # Add the benign set.
        datasets = [torch_datasets.CIFAR10(self.root, False, self.transform())]

        # Add the evil sets.
        for distortion in CIFAR10C.distortions_list:
            d = CIFAR10C(
                root=self.root, transform=self.transform(), distortion=distortion
            )
            datasets.extend(d.lvl_subsets())

        self.test_dataset = datasets
