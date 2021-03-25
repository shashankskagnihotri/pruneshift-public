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


class ImageFolderSubset(ImageFolder):
    """Selects a subset of an ImageFolder relative to the classes of a different subset.
    This is done by overwriting the _find_classes method of the ImageFolder class.

    Args:
        root: The root to the ImageFolder, where want the subset from.
        subset_root: The root which contains the classes for the subset.
        transform: The transformation that should be applied to the samples.
        target_transform: The transformation that should be applied to the targets.

    Attributes:
        classes_subset: The classes that are used.
    """

    def __init__(self, root, subset: ImageFolder, transform=None, target_transform=None):
        self.classes_subset = set(subset.classes)

        super(ImageFolderSubset, self).__init__(
            root=root, transform=transform, target_transform=target_transform
        )

    def _find_classes(self, root: str):
        """ Overwrites find classes method to remove all the labels not needed."""
        classes = set(p.stem for p in Path(self.root).iterdir())
        # Filter out all classes that are not in the subset.
        classes = set.intersection(self.classes_subset, classes)
        classes = sorted(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class TransformWrapper(Dataset):
    def __init__(self, dataset, transform, with_idx=False):
        self.dataset = dataset
        self.transform = transform
        self.with_idx = with_idx

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.with_idx:
            return idx, x, y
        return x, y

    def __len__(self):
        return len(self.dataset)


class CRDWrapper(Dataset):
    def __init__(self, dataset, transform, with_idx=False):
        self.dataset = dataset
        self.transform = transform
        self.with_idx = with_idx
        self.k = 16384
        self.percent = 1.0
        self.mode = 'exact'
        
        
        current_dataset = dataset
        while hasattr(current_dataset, 'dataset'):
            current_dataset = current_dataset.dataset
        label = current_dataset.targets
        num_classes = len(set(label))
        num_samples = len(dataset)
        
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < self.percent < 1:
            n = int(len(self.cls_negative[0]) * self.percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        target = y
        
        if self.mode == 'exact':
            pos_idx = idx
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]

        #print('pos_idx', pos_idx)
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        if self.transform is not None:
            x = self.transform(x)

        if self.with_idx:
            return idx, x, y, sample_idx
        return x, y

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


class ImageNetC:
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
        self.root = Path(root) / "corrupted"
        self.distortion = distortion
        self.transform = transform

    def lvl_subsets(self):
        path = self.root / self.distortion
        return [ImageFolder(path / str(i), self.transform) for i in range(1, 6)]
