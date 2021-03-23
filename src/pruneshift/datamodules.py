""" Provides the data modules we need for our experiments."""
import logging
from typing import Optional
from typing import Union
from pathlib import Path
from itertools import product

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import torchvision.datasets as torch_datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import CIFAR10C
from .datasets import CIFAR100C
from .datasets import ImageNetC
from .datasets import TransformWrapper
from .datasets import SplitImageFolder
from .datasets import ImageFolderSubset
from augmix.dataset import AugMixWrapper

logger = logging.getLogger(__name__)


def compose(*ts):
    """Composes multiple transformations together but filters out None's"""
    return transforms.Compose([t for t in ts if t is not None])


class ShiftDataModule(pl.LightningDataModule):
    """A clean DataModule that provides standard training sets, augmentations
    and different test sets for distribution shifts.

    Args:
        name: Name of the dataset/datamodule.
        root: Where to save/find the data.
        batch_size: The batch size used for the datamodule.
        num_workers: Number of workers used for the dataloaders.
        augmix: If true creates two additonal augmix samples,
            when "no_jsd" it creates only one augmented sample.
        deepaugment_path: Path to the deepaugment dataset that can
            be downloaded from the following repository:
            https://github.com/hendrycks/imagenet-r
        test_train: Whether to add the train_dataset to the test set.
        test_corrupted: Whether to test corruptions. 
        test_renditions: Whether to test renditions.
        with_normalize: Whether to normalize the samples, this is
            helpful for visualizing.
        only_test_transform: Whether to use only the test transformation
            for the training pipeline.
    """
    def __init__(
        self,
        name: str,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        augmix: Union[bool, str] = False,
        deepaugment_path: str = None,
        test_train: bool = False,
        test_corrupted: bool = True,
        test_renditions: bool = False,
        with_normalize: bool = True,
        only_test_transform: bool = False,
    ):
        super(ShiftDataModule, self).__init__()
        self.name = name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmix = augmix
        self.deepaugment_path = deepaugment_path
        self.test_train = test_train
        self.test_corrupted = test_corrupted
        self.test_renditions = test_renditions
        self.with_normalize = with_normalize
        self.only_test_transform = only_test_transform

        # Create the transformations corresponding to the dataset.
        self.train_transform = None
        self.test_transform = None
        self.normalize = None
        self.test_resolution = None
        self.set_transforms()

        # These are created by the setup method.
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def create_dataset(self, train: bool):
        # TODO: Add the possibility of a val_split again.
        if self.name == "cifar10":
            return torch_datasets.CIFAR10(self.root, train=train)
        elif self.name == "cifar100":
            return torch_datasets.CIFAR100(self.root, train=train)
        elif self.name == "imagenet":
            if train:
                return ImageFolder(Path(self.root) / "train")
            return ImageFolder(Path(self.root) / "val")
        raise NotImplementedError

    @property
    def corr_dataset_cls(self):
        class_map = {"cifar10": CIFAR10C, "cifar100": CIFAR100C, "imagenet": ImageNetC}
        return class_map[self.name]

    def create_corrupted_datasets(self):
        datasets = []

        for distortion in self.corr_dataset_cls.distortions_list:
            d = self.corr_dataset_cls(self.root, distortion)
            datasets.extend(d.lvl_subsets())

        return datasets

    def create_rendition_dataset(self):
        assert self.name == "imagenet"
        return ImageFolder(Path(self.root) / "renditions")

    def create_deepaugment_dataset(self, train_dataset):
        assert self.name == "imagenet"
        cae_root = Path(self.deepaugment_path) / "CAE"
        edsr_root = Path(self.deepaugment_path) / "EDSR"

        # Add the deepaugment datasets to the training dataset.
        cae_dataset = ImageFolderSubset(cae_root, train_dataset)
        edsr_dataset = ImageFolderSubset(edsr_root, train_dataset)

        return ConcatDataset([train_dataset, cae_dataset, edsr_dataset])

    def prepare_data(self):
        """Downloads datasets if possible."""

        if self.name == "cifar10":
            torch_datasets.CIFAR10(self.root, train=True, download=True)
            torch_datasets.CIFAR10(self.root, train=False, download=True)

            if self.test_corrupted:
                CIFAR10C(self.root, "snow", download=True)

        elif self.name == "cifar100":
            torch_datasets.CIFAR100(self.root, train=True, download=True)
            torch_datasets.CIFAR100(self.root, train=False, download=True)

            if self.test_corrupted:
                CIFAR100C(self.root, "snow", download=True)

    def set_transforms(self):
        """Creates the corresponding transforms."""

        # 1. Create the correct preprocessing transforms and 
        #    set the test resolutions.
        self.train_transform, self.test_transform = None, None

        if self.name == "cifar10" or self.name == "cifar100":
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            self.test_resolution = (3, 32, 32)

        else:  # self.name == "imagenet":
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]
            )
            self.test_resolution = (3, 224, 224)

        if self.only_test_transform:
            self.train_transform = self.test_transform

        # 2. Create the correct normalizations.
        if self.name == "cifar10":
            mean = (0.491, 0.4822, 0.4465)
            std = (0.247, 0.243, 0.262)
        elif self.name == "cifar100":
            # mean = (0.5071, 0.4867, 0.4408)
            # std = (0.2657, 0.2565, 0.2761)
            # This is used by augmix.
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        else:  # self.name == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    @property
    def labels(self):
        """ Provides labels for the test sets."""
        labels = ["clean"]

        if self.test_corrupted:
            for corruption in self.corr_dataset_cls.distortions_list:
                for lvl in range(1, 6):
                    labels.append(f"{corruption}_{lvl}")
        if self.test_renditions:
            labels.append("rendition")
        if self.test_train:
            labels.append("train")

        return labels

    def setup(self, stage):
        if stage == "fit":
            # 1. Create training dataset.
            train_dataset = self.create_dataset(True)

            # 2. Potentially add deepaugment samples.
            if self.deepaugment_path is not None:
                train_dataset = self.create_deepaugment_dataset(train_dataset)

            # 3. Here we potentially could split the dataset.
            val_dataset = self.create_dataset(False)

            # 4. Add standard augmentation or augmix to the training set.
            if not self.augmix:
                comb = compose(self.train_transform, self.normalize)
                train_dataset = TransformWrapper(train_dataset, comb, with_idx=True)
            else:
                train_dataset = TransformWrapper(train_dataset, self.train_transform)
                no_jsd = True if self.augmix == "no_jsd" else False
                train_dataset = AugMixWrapper(train_dataset, self.normalize, no_jsd)
                # Add indices to the samples.
                train_dataset = TransformWrapper(train_dataset, None, with_idx=True)

            # 5. Set the training and validation set.
            self.train_dataset = train_dataset

            comb = compose(self.test_transform, self.normalize)
            self.val_dataset = TransformWrapper(val_dataset, comb, with_idx=True)

        if stage == "test":
            test_datasets = []
            # 1. Add the standard test_dataset.
            test_datasets.append(self.create_dataset(False))

            # 2. Potentially add the corrupted samples.
            if self.test_corrupted:
                test_datasets.extend(self.create_corrupted_datasets())

            # 3. Potentially add the renditions samples.
            if self.test_renditions:
                test_datasets.append(self.create_rendition_dataset())

            # 4. Add the correct transformation.
            comb = compose(self.test_transform, self.normalize)
            test_datasets = [TransformWrapper(td, comb, with_idx=True) for td in test_datasets]

            # 5. Potentially add the train_dataset to the test set.
            if self.test_train:
                comb = compose(self.train_transform, self.normalize)
                train_dataset = TransformWrapper(self.create_dataset(True), comb, with_idx=True)
                test_datasets.append(train_dataset)
            
            self.test_datasets = test_datasets

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
        return self._create_dataloader(self.test_datasets)



# def datamodule(
#     name: str,
#     root: str,
#     batch_size: int = 32,
#     num_workers: int = 5,
#     val_split: Optional[float] = None,
#     val_swap: bool = False,
#     only_test_transforms: bool = False,
#     **kwargs,
# ) -> pl.LightningDataModule:
#     """Creates a LightningDataModule.
#     Args:
#         name: Name of the dataset/datamodule.
#         root: Where to save/find the data.
#         batch_size: The batch size used for the datamodule.
#         num_workers: Number of workers used for the dataloaders.
#         val_split: Amount of samples that should be used for the validation set.
#             If None the test set becomes the validation set.
#         val_swap: Whether to swap the train with the val set.
#         only_test_transforms: Whether to turn off special training transforms.
# 
#     Returns:
#         The corresponding LightningDataModule.
#     """
#     logger.info(f"Creating datamodule {name}.")
#     return BaseDataModule.subclasses[name](
#         root=root,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         val_split=val_split,
#         val_swap=val_swap,
#         only_test_transforms=only_test_transforms,
#         **kwargs,
#     )
# 
# 
# def prepare_data(name: str, root: str):
#     if name == "cifar10":
#         torch_datasets.CIFAR10(root, train=True, download=True)
#         torch_datasets.CIFAR10(root, train=False, download=True)
#     elif name == "cifar100":
#         torch_datasets.CIFAR100(root, train=True, download=True)
#         torch_datasets.CIFAR100(root, train=False, download=True)
#     elif name == "imagenet":
#         return
#     else:
#         raise NotImplementedError
# 
# 
# class BaseDataModule(pl.LightningDataModule):
#     subclasses = {}
#     mean = None
#     std = None
# 
#     def __init__(
#         self,
#         root: str,
#         batch_size: int,
#         num_workers: int,
#         val_split: Optional[float],
#         val_swap: bool = False,
#         with_normalize: bool = True,
#         only_test_transforms: bool = False,
#         **kwargs,
#     ):
#         super(BaseDataModule, self).__init__()
#         self.root = root
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.val_split = val_split
#         self.val_swap = val_swap
#         self.only_test_transforms = only_test_transforms
#         self.with_normalize = with_normalize
#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None
# 
#     def __init_subclass__(cls, **kwargs):
#         # Adds subclasses to the module.
#         super().__init_subclass__(**kwargs)
#         cls.subclasses[cls.name] = cls
# 
#     @property
#     def labels(self):
#         return ["clean"]
# 
#     def create_dataset(self, stage: str, transform=None):
#         raise NotImplementedError
# 
#     def normalizer(self):
#         ts = [transforms.ToTensor()]
#         if self.with_normalize:
#             ts.append(transforms.Normalize(self.mean, self.std))
# 
#         return transforms.Compose(ts)
# 
#     def preprocessor(self, train: bool = True):
#         raise NotImplementedError
# 
#     def setup(self, stage: str):
#         if stage == "fit":
#             transform = [self.preprocessor(), self.preprocessor(False)]
#             train_dataset, val_dataset = self.create_dataset("fit", transform)
#             self.train_dataset = TransformWrapper(
#                 train_dataset, self.normalizer(), with_idx=True
#             )
#             self.val_dataset = TransformWrapper(
#                 val_dataset, self.normalizer(), with_idx=True
#             )
# 
#         if stage == "test":
#             test_datasets = self.create_dataset("test", self.preprocessor(False))
#             assert isinstance(test_datasets, list)
#             self.test_dataset = [
#                 TransformWrapper(d, self.normalizer(), with_idx=True)
#                 for d in test_datasets
#             ]
# 
#     def _create_dataloader(self, dataset, train=False):
#         if isinstance(dataset, (tuple, list)):
#             # If not a dataset we assume a list/tuple of datasets.
#             return [self._create_dataloader(d) for d in dataset]
# 
#         return DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             shuffle=train,
#         )
# 
#     def train_dataloader(self):
#         return self._create_dataloader(self.train_dataset, True)
# 
#     def val_dataloader(self):
#         return self._create_dataloader(self.val_dataset)
# 
#     def test_dataloader(self):
#         return self._create_dataloader(self.test_dataset)
# 
# 
# class AugmixDataModule(BaseDataModule):
#     def __init__(self, no_jsd=False, **kwargs):
#         super(AugmixDataModule, self).__init__(**kwargs)
#         self.no_jsd = no_jsd
# 
#     def create_dataset(self, mode: str, transform=None):
#         if mode == "test":
#             return super(AugmixDataModule, self).create_dataset(mode, transform)
# 
#         datasets = super(AugmixDataModule, self).create_dataset(mode, transform)
#         train_dataset = AugMixWrapper(
#             datasets[0], self.normalizer(), no_jsd=self.no_jsd
#         )
# 
#         return train_dataset, datasets[1]
# 
#     def setup(self, stage: str):
#         if stage == "fit":
#             transform = [self.preprocessor(), self.preprocessor(False)]
#             train_dataset, val_dataset = self.create_dataset("fit", transform)
#             # Only difference, we do not need to normalize as this is done by the augment wrapper!
#             self.train_dataset = TransformWrapper(train_dataset, None, with_idx=True)
#             self.val_dataset = TransformWrapper(
#                 val_dataset, self.normalizer(), with_idx=True
#             )
# 
#         if stage == "test":
#             test_datasets = self.create_dataset("test", self.preprocessor(False))
#             assert isinstance(test_datasets, list)
#             self.test_dataset = [
#                 TransformWrapper(d, self.normalizer(), with_idx=True)
#                 for d in test_datasets
#             ]
# 
# 
# class CorruptedDataModule(BaseDataModule):
#     def __init__(self, lvls=None, **kwargs):
#         super(CorruptedDataModule, self).__init__(**kwargs)
# 
#         self.lvls = range(1, 6) if lvls is None else lvls
# 
#     def prepare_data(self):
#         self.corr_dataset_cls(self.root, "snow", download=True)
# 
#         super(CorruptedDataModule, self).prepare_data()
# 
#     @property
#     def labels(self):
#         """Returns labels of the datasets currently in use."""
#         labels = ["clean"]
#         for distortion in self.corr_dataset_cls.distortions_list:
#             for lvl in self.lvls:
#                 labels.append("{}_{}".format(distortion, lvl))
#         return labels
# 
#     def create_dataset(self, stage: str, transform=None):
#         if stage == "fit":
#             return super(CorruptedDataModule, self).create_dataset(stage, transform)
# 
#         datasets = super(CorruptedDataModule, self).create_dataset(stage, transform)
# 
#         for distortion in self.corr_dataset_cls.distortions_list:
#             d = self.corr_dataset_cls(self.root, distortion, transform)
#             d_lvls = d.lvl_subsets()
#             for lvl in self.lvls:
#                 datasets.append(d_lvls[lvl - 1])
# 
#         return datasets
# 
# 
# class CIFAR10Module(BaseDataModule):
#     name = "cifar10"
#     mean = (0.491, 0.4822, 0.4465)
#     std = (0.247, 0.243, 0.262)
#     cifar_cls = torch_datasets.CIFAR10
# 
#     def prepare_data(self):
#         self.cifar_cls(self.root, train=True, download=True)
#         self.cifar_cls(self.root, train=False, download=True)
# 
#     def create_dataset(self, stage: str, transform=None):
#         if stage == "fit":
#             if self.val_split is not None:
#                 dataset = self.cifar_cls(self.root, True)
#                 split_point = int(len(dataset) * self.val_split)
#                 val_dataset = Subset(dataset, range(split_point))
#                 train_dataset = Subset(dataset, range(split_point, len(dataset)))
#                 if self.val_swap:
#                     train_dataset = TransformWrapper(val_dataset, transform[0])
#                     val_dataset = TransformWrapper(train_dataset, transform[1])
#                 else:
#                     train_dataset = TransformWrapper(train_dataset, transform[0])
#                     val_dataset = TransformWrapper(val_dataset, transform[1])
#                 return train_dataset, val_dataset
#             else:
#                 train_dataset = self.cifar_cls(self.root, True, transform[0])
#                 val_dataset = self.cifar_cls(self.root, False, transform[1])
#                 return train_dataset, val_dataset
# 
#         return [self.cifar_cls(self.root, False, transform)]
# 
#     def preprocessor(self, train: bool = True):
#         if not train or self.only_test_transforms:
#             return None
# 
#         return transforms.Compose(
#             [
#                 transforms.RandomCrop(size=32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#             ]
#         )
# 
# 
# class CIFAR100Module(CIFAR10Module):
#     name = "cifar100"
#     mean = (0.5071, 0.4867, 0.4408)
#     std = (0.2657, 0.2565, 0.2761)
#     cifar_cls = torch_datasets.CIFAR100
# 
# 
# class ImageNetModule(BaseDataModule):
#     name = "imagenet"
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
# 
#     def prepare_data(self):
#         pass
# 
#     def create_dataset(self, stage: str, transform=None):
#         if stage == "fit":
#             if self.val_split is not None:
#                 dataset = SplitImageFolder(Path(self.root) / "train")
#                 val_dataset, train_dataset = dataset.split(self.val_split)
#                 train_dataset = TransformWrapper(train_dataset, transform[0])
#                 val_dataset = TransformWrapper(val_dataset, transform[1])
#                 return train_dataset, val_dataset
#             else:
#                 train_dataset = ImageFolder(
#                     Path(self.root) / "train", transform=transform[0]
#                 )
#                 val_dataset = ImageFolder(
#                     Path(self.root) / "val", transform=transform[1]
#                 )
#                 return train_dataset, val_dataset
# 
#         return [ImageFolder(Path(self.root) / "val", transform)]
# 
#     def preprocessor(self, train: bool = True):
#         ts = []
# 
#         if train and not self.only_test_transforms:
#             ts.append(transforms.RandomResizedCrop(224))
#             ts.append(transforms.RandomHorizontalFlip())
#         else:
#             ts.append(transforms.Resize(256))
#             ts.append(transforms.CenterCrop(224))
# 
#         return transforms.Compose(ts)
# 
# 
# class DeepNetModule(ImageNetModule):
#     name = "deepnet"
# 
#     def __init__(self, deepaugment_root: str, **kwargs):
#         super(DeepNetModule, self).__init__(**kwargs)
#         self.cae_root = Path(deepaugment_root) / "CAE"
#         self.edsr_root = Path(deepaugment_root) / "EDSR"
# 
#     def create_dataset(self, stage: str, transform=None):
#         if stage == "fit":
#             if self.val_split is not None:
#                 # Maybe we can delete this.
#                 raise NotImplementedError
# 
#             train_dataset, val_dataset = super(DeepNetModule, self).create_dataset(
#                 stage, transform
#             )
# 
#             # Add the deepaugment datasets to the training dataset.
#             cae_dataset = ImageFolderSubset(self.cae_root, train_dataset, transform[0])
#             edsr_dataset = ImageFolderSubset(
#                 self.edsr_root, train_dataset, transform[0]
#             )
# 
#             train_dataset = ConcatDataset([train_dataset, cae_dataset, edsr_dataset])
#             return train_dataset, val_dataset
# 
#         return super(DeepNetModule, self).create_dataset(stage, transform)
# 
# 
# class ImageNetRenditionModule(ImageNetModule):
#     name = "imagenet_renditions"
# 
#     def create_dataset(self, stage: str, transform=None):
#         if stage == "fit":
#             return super(ImageNetRenditionModule, self).create_dataset(stage, transform)
# 
#         datasets = super(ImageNetRenditionModule, self).create_dataset(stage, transform)
# 
#         datasets.append(ImageFolder(Path(self.root) / "renditions", transform))
# 
#         return datasets
# 
#     @property
#     def labels(self):
#         return ["clean", "rendition"]
# 
# 
# class CIFAR10AugmixModule(AugmixDataModule, CIFAR10Module):
#     name = "cifar10_augmix"
# 
# 
# class CIFAR100AugmixModule(AugmixDataModule, CIFAR100Module):
#     name = "cifar100_augmix"
# 
# 
# class ImageNetAugmixModule(AugmixDataModule, ImageNetModule):
#     name = "imagenet_augmix"
# 
# 
# class DeepNetAugmixModule(AugmixDataModule, DeepNetModule):
#     name = "deepnet_augmix"
# 
# 
# class CIFAR10CModule(CorruptedDataModule, CIFAR10Module):
#     name = "cifar10_corrupted"
#     corr_dataset_cls = CIFAR10C
# 
# 
# class CIFAR100CModule(CorruptedDataModule, CIFAR100Module):
#     name = "cifar100_corrupted"
#     corr_dataset_cls = CIFAR100C
# 
# 
# class ImageNetCModule(CorruptedDataModule, ImageNetModule):
#     name = "imagenet_corrupted"
#     corr_dataset_cls = ImageNetC
# 
# 
# class CIFAR10AugmixCModule(AugmixDataModule, CIFAR10CModule):
#     name = "cifar10_augmix_corrupted"
# 
# 
# class CIFAR100AugmixCModule(AugmixDataModule, CIFAR100CModule):
#     name = "cifar100_augmix_corrupted"
# 
# 
# class ImageNetAugmixCModule(AugmixDataModule, ImageNetCModule):
#     name = "imagenet_augmix_corrupted"
