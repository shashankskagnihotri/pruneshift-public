import os
from urllib.request import urlopen

import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset


class ExternalDataset(Dataset):
    dir_name: str = None
    url: str = None

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        if not self.exists:
            self.download()

    @property
    def path(self):
        return os.path.join(self.root, self.dir_name)

    @property
    def exists(self):
        return os.path.isdir(self.path)

    def download(self):
        download_and_extract_archive(url=self.url,
                                     download_root=self.root,
                                     remove_finished=True)


class CIFAR10C(ExternalDataset):
    dir_name = "CIFAR-10-C"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"

    def __init__(self, root: str, distortion: str, transform=None,
                 lvl: int = 1):
        super(CIFAR10C, self).__init__(root, transform)
        assert 1 <= lvl <= 5
        self.distortion = distortion
        self.lvl = lvl
        self._images = np.load(self.image_path)
        self._labels = np.load(self.label_path)

    @property
    def image_path(self):
        return os.path.join(self.path, self.distortion + ".npy")

    @property
    def label_path(self):
        return os.path.join(self.path, "labels.npy")

    def _lvl_slice(self):
        return slice(10000 * self.lvl, 10000 * (self.lvl + 1))

    def __getitem__(self, idx):
        img = self._images[self._lvl_slice()][idx]
        lbl = self._labels[self._lvl_slice()][idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl 

    def __len__(self):
        return 10000
