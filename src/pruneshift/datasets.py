import os
from pathlib import Path

from typing import Tuple
from typing import Optional
from typing import Callable
import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch.utils.data as data

from augmix.dataset import AugMixWrapper


class SplitImageFolder(ImageFolder):
    """Efficiently calculates split sets from image folders.

    Args:
        root: Location of the image folder.
        transform: Transformation for the images.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super(SplitImageFolder, self).__init__(root=root, transform=transform)

    def split(self, val_split: float) -> Tuple[Dataset, Dataset]:
        """Splits the dataset into a validation and training dataset.

        Args:
            val_split: In [0, 1] the amount of samples for the validation set.

        Returns:
            val_dataset, train_dataset
        """
        assert 0 <= val_split <= 1

        val_ranges = []
        train_ranges = []

        for cr in self.class_ranges():
            split_point = int(len(cr) * val_split)
            val_ranges.append(cr[:split_point])
            train_ranges.append(cr[split_point:])

        val_idx = [idx for part in val_ranges for idx in part]
        train_idx = [idx for part in train_ranges for idx in part]

        return data.Subset(self, val_idx), data.Subset(self, train_idx)

    def class_ranges(self):
        class_ranges = []
        last_border = 0

        for border in np.cumsum(self.num_examples()):
            class_ranges.append(range(last_border, border))
            last_border = border

        return class_ranges

    def num_examples(self):
        path = Path(self.root)

        # We need to sort the class directories as this is done by ImageFolder.
        sorted_class_dirs = sorted(path.iterdir())
        num_examples = [len(list(d.glob("*.*"))) for d in sorted_class_dirs]

        return np.array(num_examples)


class TransformWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        if self.transform is None:
            return self.dataset[idx]
        x, y = self.dataset[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)


class ExternalDataset(Dataset):
    dir_name: str = None
    url: str = None

    def __init__(self, root: str, transform=None, download=False):
        self.root = root
        self.transform = transform
        if not self.exists and download:
            self.download()

    @property
    def path(self):
        return os.path.join(self.root, self.dir_name)

    @property
    def exists(self):
        return os.path.isdir(self.path)

    def download(self):
        download_and_extract_archive(
            url=self.url, download_root=self.root, remove_finished=True
        )


class CIFAR10C(ExternalDataset):
    dir_name = "CIFAR-10-C"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    distortions_list = [
        "shot_noise",
        "gaussian_noise",
        "saturate",
        "jpeg_compression",
        "frost",
        "defocus_blur",
        "zoom_blur",
        "snow",
        "motion_blur",
        "speckle_noise",
        "gaussian_blur",
        "brightness",
        "pixelate",
        "impulse_noise",
        "fog",
        "contrast",
        "glass_blur",
        "spatter",
        "elastic_transform",
    ]

    def __init__(self, root: str, distortion: str, transform=None, download=False):
        super(CIFAR10C, self).__init__(root, transform, download)
        self.distortion = distortion
        self._images = np.load(self.image_path)
        self._labels = np.load(self.label_path)

    @property
    def image_path(self):
        return os.path.join(self.path, self.distortion + ".npy")

    @property
    def label_path(self):
        return os.path.join(self.path, "labels.npy")

    def lvl_subsets(self):
        """Splits the dataset into the different severity lvls."""
        ranges = [range(10000 * l, 10000 * (l + 1)) for l in range(5)]
        return [data.Subset(self, r) for r in ranges]

    def __getitem__(self, idx):
        img = self._images[idx]
        lbl = self._labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return 50000


class CIFAR100C(CIFAR10C):
    dir_name = "CIFAR-100-C"
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"


class ImageNet100C:
    distortions_list = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "shot_noise",
        "snow",
        "zoom_blur",
    ]

    def __init__(self, root: str, distortion: str, transform=None, download=False):
        self.root = root
        self.distortion = distortion
        self.transform = transform

    def lvl_subsets(self):
        path = Path(self.root) / self.distortion
        return [ImageFolder(path / str(i), self.transform) for i in range(1, 6)]

