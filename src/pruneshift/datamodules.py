""" Provides the data modules we need for our experiments."""

import logging
from typing import Optional
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision.datasets as torch_datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import CIFAR10C
from .datasets import CIFAR100C
from .datasets import ImageNetC
from .datasets import TransformWrapper
from .datasets import SplitImageFolder
from augmix.dataset import AugMixWrapper


logger = logging.getLogger(__name__)


def datamodule(
    name: str,
    root: str,
    batch_size: int = 32,
    num_workers: int = 5,
    val_split: Optional[float] = None,
    val_swap: bool = False,
    **kwargs,
) -> pl.LightningDataModule:
    """Creates a LightningDataModule.
    Args:
        name: Name of the dataset/datamodule.
        root: Where to save/find the data.
        batch_size: The batch size used for the datamodule.
        num_workers: Number of workers used for the dataloaders.
        val_split: Amount of samples that should be used for the validation set.
            If None the test set becomes the validation set.
        val_swap: Whether to swap the train with the val set.

    Returns:
        The corresponding LightningDataModule.
    """
    logger.info(f"Creating datamodule {name}.")
    return BaseDataModule.subclasses[name](
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        val_swap=val_swap,
        **kwargs,
    )


class BaseDataModule(pl.LightningDataModule):
    subclasses = {}
    mean = None
    std = None

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        val_split: Optional[float],
        val_swap: bool = False,
        with_normalize: bool = True,
        **kwargs,
    ):
        super(BaseDataModule, self).__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.val_swap = val_swap
        self.with_normalize = with_normalize
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def __init_subclass__(cls, **kwargs):
        # Adds subclasses to the module.
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.name] = cls

    @property
    def labels(self):
        return ["clean"]

    def create_dataset(self, stage: str, transform=None):
        raise NotImplementedError

    def normalizer(self):
        ts = [transforms.ToTensor()]
        if self.with_normalize:
            ts.append(transforms.Normalize(self.mean, self.std))

        return transforms.Compose(ts)

    def preprocessor(self, train: bool = True):
        raise NotImplementedError

    def setup(self, stage: str):
        if stage == "fit":
            transform = [self.preprocessor(), self.preprocessor(False)]
            train_dataset, val_dataset = self.create_dataset("fit", transform)
            self.train_dataset = TransformWrapper(train_dataset, self.normalizer(), with_idx=True)
            self.val_dataset = TransformWrapper(val_dataset, self.normalizer(), with_idx=True)

        if stage == "test":
            test_datasets = self.create_dataset("test", self.preprocessor(False))
            assert isinstance(test_datasets, list)
            self.test_dataset = [
                TransformWrapper(d, self.normalizer(), with_idx=True) for d in test_datasets
            ]

    def _create_dataloader(self, dataset, train=False):
        if isinstance(dataset, (tuple, list)):
            # If not a dataset we assume a list/tuple of datasets.
            return [self._create_dataloader(d) for d in dataset]

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=train,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)


class AugmixDataModule(BaseDataModule):

    def __init__(self, no_jsd=False, **kwargs):
        super(AugmixDataModule, self).__init__(**kwargs)
        self.no_jsd = no_jsd

    def create_dataset(self, mode: str, transform=None):
        if mode == "test":
            return super(AugmixDataModule, self).create_dataset(mode, transform)

        datasets = super(AugmixDataModule, self).create_dataset(mode, transform)
        train_dataset = AugMixWrapper(
            datasets[0], self.normalizer(), no_jsd=self.no_jsd
        )

        return train_dataset, datasets[1]

    def setup(self, stage: str):
        if stage == "fit":
            transform = [self.preprocessor(), self.preprocessor(False)]
            train_dataset, val_dataset = self.create_dataset("fit", transform)
            # Only difference, we do not need to normalize as this is done by the augment wrapper!
            self.train_dataset = TransformWrapper(train_dataset, None, with_idx=True)
            self.val_dataset = TransformWrapper(val_dataset, self.normalizer(), with_idx=True)

        if stage == "test":
            test_datasets = self.create_dataset("test", self.preprocessor(False))
            assert isinstance(test_datasets, list)
            self.test_dataset = [
                TransformWrapper(d, self.normalizer(), with_idx=True) for d in test_datasets
            ]


class CorruptedDataModule(BaseDataModule):

    def __init__(self, lvls=None, **kwargs):
        super(CorruptedDataModule, self).__init__(**kwargs)

        self.lvls = range(1, 6) if lvls is None else lvls

    def prepare_data(self):
        self.corr_dataset_cls(self.root, "snow", download=True)

        super(CorruptedDataModule, self).prepare_data()

    @property
    def labels(self):
        """Returns labels of the datasets currently in use."""
        labels = ["clean"]
        for distortion in self.corr_dataset_cls.distortions_list:
            for lvl in self.lvls:
                labels.append("{}_{}".format(distortion, lvl))
        return labels

    def create_dataset(self, stage: str, transform=None):
        if stage == "fit":
            return super(CorruptedDataModule, self).create_dataset(stage, transform)

        datasets = super(CorruptedDataModule, self).create_dataset(stage, transform)

        for distortion in self.corr_dataset_cls.distortions_list:
            d = self.corr_dataset_cls(self.root, distortion, transform)
            d_lvls = d.lvl_subsets()
            for lvl in self.lvls:
                datasets.append(d_lvls[lvl - 1])

        return datasets


class CIFAR10Module(BaseDataModule):
    name = "cifar10"
    mean = (0.491, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.262)
    cifar_cls = torch_datasets.CIFAR10

    def prepare_data(self):
        self.cifar_cls(self.root, train=True, download=True)
        self.cifar_cls(self.root, train=False, download=True)

    def create_dataset(self, stage: str, transform=None):
        if stage == "fit":
            if self.val_split is not None:
                dataset = self.cifar_cls(self.root, True)
                split_point = int(len(dataset) * self.val_split)
                val_dataset = Subset(dataset, range(split_point))
                train_dataset = Subset(dataset, range(split_point, len(dataset)))
                if self.val_swap:
                    train_dataset = TransformWrapper(val_dataset, transform[0])
                    val_dataset = TransformWrapper(train_dataset, transform[1])
                else:
                    train_dataset = TransformWrapper(train_dataset, transform[0])
                    val_dataset = TransformWrapper(val_dataset, transform[1])
                return train_dataset, val_dataset
            else:
                train_dataset = self.cifar_cls(self.root, True, transform[0])
                val_dataset = self.cifar_cls(self.root, False, transform[1])
                return train_dataset, val_dataset

        return [self.cifar_cls(self.root, False, transform)]

    def preprocessor(self, train: bool = True):
        if not train:
            return None

        return transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )


class CIFAR100Module(CIFAR10Module):
    name = "cifar100"
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2657, 0.2565, 0.2761)
    cifar_cls = torch_datasets.CIFAR100


class ImageNetModule(BaseDataModule):
    name = "imagenet"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def prepare_data(self):
        pass

    def create_dataset(self, stage: str, transform=None):
        if stage == "fit":
            if self.val_split is not None:
                dataset = SplitImageFolder(Path(self.root) / "train")
                val_dataset, train_dataset = dataset.split(self.val_split)
                train_dataset = TransformWrapper(train_dataset, transform[0])
                val_dataset = TransformWrapper(val_dataset, transform[1])
                return train_dataset, val_dataset
            else:
                train_dataset = ImageFolder(
                    Path(self.root) / "train", transform=transform[0]
                )
                val_dataset = ImageFolder(
                    Path(self.root) / "val", transform=transform[1]
                )
                return train_dataset, val_dataset

        return [ImageFolder(Path(self.root) / "val", transform)]

    def preprocessor(self, train: bool = True):
        ts = []

        if train:
            ts.append(transforms.RandomResizedCrop(224))
            ts.append(transforms.RandomHorizontalFlip())
        else:
            ts.append(transforms.Resize(256))
            ts.append(transforms.CenterCrop(224))

        return transforms.Compose(ts)


class ImageNetRenditionModule(ImageNetModule):
    name = "imagenet_renditions"

    def create_dataset(self, stage: str, transform=None):
        if stage == "fit":
            return super(ImageNetRenditionModule, self).create_dataset(stage, transform)

        datasets = super(ImageNetRenditionModule, self).create_dataset(stage, transform)

        datasets.append(ImageFolder(Path(self.root) / "renditions", transform))

        return datasets

    @property
    def labels(self):
        return ["clean", "rendition"]


class CIFAR10AugmixModule(AugmixDataModule, CIFAR10Module):
    name = "cifar10_augmix"


class CIFAR100AugmixModule(AugmixDataModule, CIFAR100Module):
    name = "cifar100_augmix"


class ImageNetAugmixModule(AugmixDataModule, ImageNetModule):
    name = "imagenet_augmix"


class CIFAR10CModule(CorruptedDataModule, CIFAR10Module):
    name = "cifar10_corrupted"
    corr_dataset_cls = CIFAR10C


class CIFAR100CModule(CorruptedDataModule, CIFAR100Module):
    name = "cifar100_corrupted"
    corr_dataset_cls = CIFAR100C


class ImageNetCModule(CorruptedDataModule, ImageNetModule):
    name = "imagenet_corrupted"
    corr_dataset_cls = ImageNetC


class CIFAR10AugmixCModule(AugmixDataModule, CIFAR10CModule):
    name = "cifar10_augmix_corrupted"


class CIFAR100AugmixCModule(AugmixDataModule, CIFAR100CModule):
    name = "cifar100_augmix_corrupted"


class ImageNetAugmixCModule(AugmixDataModule, ImageNetCModule):
    name = "imagenet_augmix_corrupted"
