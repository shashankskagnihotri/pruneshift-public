""" Provides the data modules we need for our experiments.

A Datamodule must be explicitly provided.

Transform Mixins

Registration of Databases


"""
import abc

from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.datasets as torch_datasets
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import CIFAR10C



def datamodule(name: str,
               root: str,
               batch_size: int = 32,
               num_workers: int = 5) -> pl.LightningDataModule:
    """Creates a LightningDataModule.

    Args:
        name: Name of the dataset/datamodule.
        root: Where to download the data if necessary.
        batch_size: The batch size used for the datamodule. 
        num_workers: Number of workers used for the dataloaders.
    Returns:
        The corresponding LightningDataModule.
    """
    return BaseDataModule.subclasses[name](root, batch_size, num_workers)



class BaseDataModule(pl.LightningDataModule):
    subclasses = {} 

    def __init__(self,
                 root: str,
                 batch_size: int,
                 num_workers: int):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def create_dataloader(self, dataset):
        if not isinstance(dataset, Dataset):
            # If not a dataset we assume a list/tuple of datasets.
            return [self._create_dataloader(d) for d in dataset]
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def __init_subclass__(cls, **kwargs):
        # Add subclasses to the module.
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def transform(self, train: bool = False):
        return transforms.ToTensor()

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)


class CIFAR10Transform:
    def transform(self, train: bool = False):
        transform = []
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)

        if train:
            transform.append(transforms.RandomCrop(size=32, padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))

        return transforms.Compose(transform)


class CIFAR10(CIFAR10Transform, BaseDataModule):
    def prepare_data(self):
        torch_datasets.CIFAR10(self.root, train=True, download=True)
        torch_datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self, stage):
        create_fn = torch_datasets.CIFAR10 
        self.testset = create_fn(self.root, transform=self.transform())
                         
        trainset = create_fn(self.root, transform=self.transform(True))
        trainset, valset = random_split(trainset, [55000, 5000])
        self.train_dataset = trainset
        self.val_dataset = valset



# TODO: Make it possible to have multiple datasets.
class CIFAR10C(CIFAR10Transform, BaseDataModule):
    def __init__(self,
                 root: str,
                 distortion: str,
                 lvl: int = 1,
                 batch_size: int = 32,
                 num_workers: int = 5):
        super().__init__(root, batch_size, num_workers)
        self.distortion = distortion
        self.lvl = lvl

    def prepare_data(self):
        # Make sure that every dataset is downloaded.
        CIFAR10C(self.root, "snow", lvl=1)

    def setup(self, stage):
        self.train_dataset = CIFAR10(self.root, transform=self.transform(True))
        test_set = CIFAR10(self.root, train=False,
                           transform=self.transform())
        # Now add the corrupted set.
        test_set_corr= CIFAR10C(root=self.root,
                                transform=self.transform(),
                                distortion=self.distortion,
                                lvl=self.lvl)
        self.test_dataset = [test_set, test_set_corr]

