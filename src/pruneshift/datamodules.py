""" Provides the data modules we need for our experiments."""
import logging
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as torch_datasets
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import CIFAR10C
from .datasets import CIFAR100C
from augmix.dataset import AugMixWrapper


logger = logging.getLogger(__name__)


def datamodule(
    name: str, root: str, batch_size: int = 32, num_workers: int = 5, **kwargs
) -> pl.LightningDataModule:
    """ Creates a LightningDataModule.
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
        super(BaseDataModule, self).__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def __init_subclass__(cls, **kwargs):
        # Adds subclasses to the module.
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.name] = cls

    def create_dataset(self, stage: str, transform=None):
        raise NotImplementedError

    def transform(self, train: bool=True):
        raise NotImplementedError

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.val_dataset = self.create_dataset("fit", self.transform())
        if stage == "test":
            self.test_dataset = self.create_dataset("test", self.transform(False))

    def _create_dataloader(self, dataset, shuffle=False):
        if isinstance(dataset, (tuple, list)):
            # If not a dataset we assume a list/tuple of datasets.
            return [self._create_dataloader(d) for d in dataset]

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        dataset = self.test_dataset
        if not isinstance(dataset, (tuple, list)):
            return self._create_dataloader([dataset])
        return self._create_dataloader(dataset)


class AugmixDataModule(BaseDataModule):

    def create_dataset(self, mode: str, transform):
        # The preprocessing transform takes place after augmix!
        datasets = super(AugmixDataModule, self).create_dataset(mode, None)

        if mode == "fit":
            return [AugMixWrapper(d, transform) for d in datasets]

        return AugMixWrapper(datasets, transform)


class CorruptedDataModule(BaseDataModule):

    corr_dataset_cls = None 

    def __init__(self, root: str, batch_size: int, num_workers: int, lvls=None):
        super(CorruptedDataModule, self).__init__(root, batch_size, num_workers)
        self.lvls = range(1, 6) if lvls is None else lvls

    def prepare_data(self):
        self.corr_dataset_cls(self.root, "snow", download=True)

        super(CorruptedDataModule, self).prepare_data()

    @property
    def labels(self):
        """Returns labels of the datasets currently in use."""
        labels = ["original"]
        for distortion in self.corr_dataset_cls.distortions_list:
            for lvl in self.lvls:
                labels.append("{}_{}".format(distortion, lvl))
        return labels

    def corrupted_datasets(self, corruption: str):
        raise NotImplementedError

    def create_dataset(self, stage: str, transform):
        if stage == "fit":
            msg = "Corrupted dataset should not be used for training"
            raise RuntimeError(msg)

        datasets = [super(CorruptedDataModule, self).create_dataset(stage, transform)]

        for distortion in self.corr_dataset_cls.distortions_list:
            d = self.corr_dataset_cls(self.root, distortion, transform)
            d_lvls = d.lvl_subsets()
            for lvl in self.lvls:
                datasets.append(d_lvls[lvl - 1])

        return datasets


class CIFAR10Module(BaseDataModule):
    name = "cifar10"

    def prepare_data(self):
        torch_datasets.CIFAR10(self.root, train=True, download=True)
        torch_datasets.CIFAR10(self.root, train=False, download=True)

    def create_dataset(self, mode: str, transform):
        if mode == "fit":
            dataset = torch_datasets.CIFAR10(self.root, True, transform)
            return random_split(dataset, [45000, 5000]) 
        return torch_datasets.CIFAR10(self.root, False, transform)

    def transform(self, train: bool = True):
        transform = []
        mean, std = (0.491, 0.4822, 0.4465), (0.247, 0.243, 0.262)

        if train:
            transform.append(transforms.RandomCrop(size=32, padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))

        return transforms.Compose(transform)


class CIFAR100Module(BaseDataModule):
    name = "cifar100"

    def prepare_data(self):
        torch_datasets.CIFAR100(self.root, train=True, download=True)
        torch_datasets.CIFAR100(self.root, train=False, download=True)

    def create_dataset(self, mode: str, transform):
        if mode == "fit":
            dataset = torch_datasets.CIFAR10(self.root, True, transform)
            return random_split(dataset, [45000, 5000]) 
        return torch_datasets.CIFAR10(self.root, False, transform)

    def transform(self, train: bool = True):
        transform = []
        mean, std = (0.5071, 0.4867, 0.4408), (0.2657, 0.2565, 0.2761)

        if train:
            transform.append(transforms.RandomCrop(size=32, padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))

        return transforms.Compose(transform)



class CIFAR10AugmixModule(AugmixDataModule, CIFAR10Module):
    name = "cifar10_augmix"


class CIFAR100AugmixModule(AugmixDataModule, CIFAR100Module):
    name = "cifar100_augmix"


class CIFAR10CModule(CorruptedDataModule, CIFAR10Module):
    name = "cifar10_corrupted"
    corr_dataset_cls = CIFAR10C


class CIFAR100CModule(CorruptedDataModule, CIFAR100Module):
    name = "cifar100_corrupted"
    corr_dataset_cls = CIFAR100C


# class ImageNetModule(BaseDataModule):
#     name = "imagenet"
# 
#     def prepare_data(self):
#         torch_datasets.ImageNet()
# 
#     def transform(self, train: bool = True):
#         transform = []
#         mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# 
#         if train:
#             transform.append(transforms.RandomResizedCrop(224))
#             transform.append(transforms.RandomHorizontalFlip())
#         transform.append(transforms.ToTensor())
#         transform.append(transforms.Normalize(mean, std))
# 
#         return transform
# 
#     def setup(self, stage: str = None) -> None:
# 
#         create_fn = torch_datasets.ImageNet
# 
#         if stage == "test" or stage is None:
#             self.test_dataset = create_fn(self.root, False,
#                                           transform=self.transform(False))
#         if stage == "fit" or stage is None:
#             trainset = create_fn(self.root, True, transform=self.transform())
#             trainset, valset = random_split(trainset, [45000, 5000])
#             self.train_dataset = trainset
#             self.val_dataset = valset
# 
