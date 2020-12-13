import os
import glob
from urllib.request import urlopen
from pathlib import Path

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
        "labels",
        "contrast",
        "glass_blur",
        "spatter",
        "elastic_transform",
    ]

    def __init__(self, root: str, distortion: str, transform=None):
        super(CIFAR10C, self).__init__(root, transform)
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
        return [torch.utils.data.Subset(self, r) for r in ranges]

    def __getitem__(self, idx):
        img = self._images[idx]
        lbl = self._labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return 50000 

