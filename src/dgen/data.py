import os
from argparse import ArgumentParser, Namespace
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from dgen.augmentation.augmented_dataset import AugmentedDataset


class TestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, loader_name=None, **kwargs):
        self.loader_name = loader_name # kwargs.pop('name')
        super().__init__(*args, **kwargs)



class ImageNetData(pl.LightningDataModule):

    CORRUPTIONS = [ 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression']

    def __init__(self, args):
        super().__init__()
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],)
        self.preprocess = transforms.Compose([transforms.ToTensor(),
                                              normalize,])
        self.workers = args.workers
        self.args = args

    def train_dataloader(self):
        train_dir = os.path.join(self.data_path, 'train')

        train_dataset = datasets.ImageFolder(
                train_dir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ]))

        train_dataset = AugmentedDataset(train_dataset,
                                         image_size = 224,
                                         preprocess = self.preprocess,
                                         args = self.args
                                         )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )
        return train_loader

    def val_dataloader(self):
        val_dir = os.path.join(self.data_path, 'val')
        val_loader = TestDataLoader(
                                datasets.ImageFolder(val_dir, transforms.Compose([
                                                              transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              self.preprocess])),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.workers,
        )
        val_loader.loader_name = 'clean'
        return val_loader

    def test_dataloader(self):
        test_loaders = [self.val_dataloader()]
        for c in self.CORRUPTIONS[:1]:
            for s in range(1, 6):
                valdir = os.path.join(self.data_path,'corrupted', c, str(s))
                val_loader = TestDataLoader(
                                        datasets.ImageFolder(valdir, self.preprocess),
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.workers,
                                        )
                val_loader.loader_name = "corr_{}_severity_{}".format(c,s)
                test_loaders.append(val_loader)
        return test_loaders

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--mixture-width',
            default=3,
            type=int,
            help='number of augmentation chains to mix per augmented example')
        parser.add_argument(
            '--mixture-depth',
            default=-1,
            type=int,
            help='depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
        parser.add_argument(
            '--aug-severity',
            default=1,
            type=int,
            help='severity of base augmentation operators')
        parser.add_argument(
            '--aug-prob-coeff',
            default=1.,
            type=float,
            help='probability distribution coefficients')
        parser.add_argument(
            '--aug-type',
            default='augmix',
            choices=['augmix', 'simple', 'none'],
            help='choose the augmentation type')
        parser.add_argument(
            '--aug-list',
            nargs="+",
            default = [],
            help='provide a list of augmentations to use')

        return parser
