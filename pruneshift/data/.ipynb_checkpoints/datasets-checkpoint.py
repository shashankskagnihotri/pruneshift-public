import os
from urllib.request import urlopen

import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset


class ExternalDataset(Dataset):
    dir_name: str = None

    def __init__(self, root_path: str):
        self.root_path = root_path
        if not self.exists:
            self.download()

    @property
    def path(self):
        return os.path.join(self.root_path, self.dir_name)

    @property
    def exists(self):
        return os.path.isdir(self.path)

    def download(self):
        download_and_extract_archive(url=self.url, download_root=self.path,
                                                    remove_finished=True)


class CIFAR10C(ExternalDataset):
    dir_name = "Cifar-10-C"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"

    def __init__(self, root_path: str, distortion: str, lvl: int = 1):
        super(CIFAR10C, self).__init__(root_path)
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
        return os.path.join(self.root_path, "labels.npy")

    def _lvl_slice(self):
        return slice(10000 * self.lvl, 10000 * (self.lvl + 1))

    def __getitem__(self, idx):
        img = self._images[self._lvl_slice][idx]
        lbl = self._labels[self._lvl_slice][idx]
        return img, lbl 

    def __len__(self):
        return 10000

