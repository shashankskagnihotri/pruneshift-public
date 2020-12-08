""" Provides the data modules we need for our experiments."""
import abc

from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import CIFAR10C


class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 batch_size: int = 32,
                 num_workers: int = 5):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_dataloader(self, dataset):
        if not isinstance(dataset, Dataset):
            # If not a dataset we assume a list/tuple of datasets.
            return [self._create_dataloader(d) for d in dataset]
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)


class CIFAR10DistrShift(BaseDataModule):
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
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)
        CIFAR10C(self.root, "snow", lvl=1)

    def transform(self, train: bool = False):
        transform = []
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)

        if train:
            transform.append(transforms.RandomCrop(size=32, padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))

        return transforms.Compose(transform)

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


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 batch_size: int = 32,
                 num_workers: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers: int = num_workers
        self.dataset_path = dataset_path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # download only
        MNIST(self.dataset_path, train=True, download=True, transform=transforms.ToTensor())
        MNIST(self.dataset_path, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = MNIST(self.dataset_path, train=True, download=False, transform=transform)
        mnist_test = MNIST(self.dataset_path, train=False, download=False, transform=transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
