import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from dgen.augmentation import augmentations


def simple_aug(image, preprocess, args):
    all_ops = augmentations.augmentations_all
    op_names = [o.__name__ for o in all_ops]
    aug_list = []
    for op_name in args.aug_list:
        op_name = op_name.strip()
        idx = op_names.index(op_name)
        aug_list.append(all_ops[idx])
    prob = 0.5
    for op in aug_list:
        if np.random.rand() > prob:
            image = op(image, args.aug_severity)
    return preprocess(image)

def augmix(image, preprocess, args):
    aug_list = augmentations.augmentations
    if args.aug_list:
        all_ops = augmentations.augmentations_all
        op_names = [o.__name__ for o in all_ops]
        aug_list = []
        for op_name in args.aug_list:
            op_name = op_name.strip()
            idx = op_names.index(op_name)
            aug_list.append(all_ops[idx])

    ws = np.float32(np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
    m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)
    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

def noaug(image, preprocess, args):
    return preprocess(image)

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_size, preprocess, args):
        augmentations.set_image_size(image_size)
        self.dataset = dataset
        self.preprocess = preprocess
        self.args = args
        self.aug_method = self._get_aug_method(args.aug_type)

    def _get_aug_method(self, aug_type):
        print("Setting aug type: {}".format(aug_type))
        if aug_type == 'augmix':
            return augmix
        elif aug_type == 'simple':
            return simple_aug
        elif aug_type == 'none':
            return noaug
        else:
            raise(ValueError("Invalid aug type"))

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.args.no_jsd:
            return self.aug_method(x, self.preprocess, self.args), y
        else:
            im_tuple = (self.preprocess(x), self.aug_method(x, self.preprocess, self.args),
                                            self.aug_method(x, self.preprocess, self.args))
        return im_tuple, y

    def __len__(self):
        return len(self.dataset)
