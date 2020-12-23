""" Provides the data modules we need for our experiments."""
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as torch_datasets
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import CIFAR10C


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

    def transform(self, train: bool = False):
        return transforms.ToTensor()

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def _create_plenty(self, dataset):
        if not isinstance(dataset, (tuple, list)):
            return self._create_dataloader([dataset])
        return self._create_dataloader(dataset)

    def val_dataloader(self):
        return self._create_plenty(self.val_dataset)

    def test_dataloader(self):
        return self._create_plenty(self.test_dataset)


class CIFAR10Module(BaseDataModule):
    def prepare_data(self):
        torch_datasets.CIFAR10(self.root, train=True, download=True)
        torch_datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self, stage: str) -> None:
        create_fn = torch_datasets.CIFAR10
        self.test_dataset = create_fn(self.root, False,
                                      transform=self.transform(False))

        trainset = create_fn(self.root, True, transform=self.transform())
        trainset, valset = random_split(trainset, [45000, 5000])
        self.train_dataset = trainset
        self.val_dataset = valset

    def transform(self, train: bool = True):
        transform = []
        mean, std = (0.491, 0.4822, 0.4465), (0.247, 0.243, 0.262)

        if train:
            transform.append(transforms.RandomCrop(size=32, padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))

        return transforms.Compose(transform)


class CIFAR10CModule(CIFAR10Module):
    def __init__(self, root: str, batch_size: int, num_workers: int, lvls=5):
        super(CIFAR10CModule, self).__init__(root, batch_size, num_workers)
        self.lvls = range(1, 6) if lvls is None else lvls
        
    @property
    def labels(self):
        """Returns labels of the datasets currently in use."""
        labels = [""]
        for distortion in CIFAR10C.distortions_list:
            for lvl in self.lvls:
                labels.append("{}_{}".format(distortion, lvl))
        return labels

    def prepare_data(self):
        # It is enough to download one
        torch_datasets.CIFAR10(self.root, train=False, download=True)
        CIFAR10C(self.root, "snow")

    def setup(self, stage):
        # Add the benign set.
        datasets = [torch_datasets.CIFAR10(self.root, False, self.transform(False))]

        # Add the evil sets.
        for distortion in CIFAR10C.distortions_list:
            d = CIFAR10C(
                root=self.root, transform=self.transform(False), distortion=distortion
            )
            d_lvls = d.lvl_subsets()
            for lvl in self.lvls:
                datasets.append(d_lvls[lvl - 1])

        self.test_dataset = datasets

