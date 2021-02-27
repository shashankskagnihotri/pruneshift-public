import augmix.augmentations
import numpy as np
import torch


def aug(
    image,
    preprocess,
    all_ops: bool = False,
    mixture_width: int = 3,
    mixture_depth: int = -1,
    aug_severity: int = 3,
):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.
      all_ops: Whether to use all operations.
      mixture_width: Number of augmentation chains to mix.
      mixture_depth: The length of the augmentation chains. 
      aug_severity: The severity of the base augmentation operators.

    Returns:
      mixed: Augmented and mixed image.
    """
    augmix.augmentations.IMAGE_SIZE = image.size[0]

    aug_list = augmix.augmentations.augmentations
    if all_ops:
        aug_list = augmix.augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixWrapper(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset: torch.utils.data.Dataset, transform,
            no_jsd: bool = False, all_ops: bool = False,
            mixture_width: int = 3, mixture_depth: int = -1, aug_severity: int = 3):
        """ Dataset wrapper to perform AugMix augmentation.

        Args:
            dataset: The dataset that should be wrapped.
            transform: The transform function, must at least convert to torch tensor.
            no_jsd: Whether to return three samples per example for the JSD consistency
                loss, or only the modified image.
            all_ops: Whether to include all operations including some ImageNetC
                operations, or only non ImageNetC operations.
            mixture_width: The number of operations chain to generate.
            mixture_depth: The length of the operations chains, when -1 the length
                is sampled from {1, 2, 3}.
            aug_severity: The severity of the base augmentation operations.

        Returns:
            The wrapped dataset.
        """
        self.dataset = dataset
        self.transform = transform 
        self.no_jsd = no_jsd
        self.all_ops = all_ops
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity

    def _aug(self, img):
        return aug(img, self.transform, self.all_ops, self.mixture_width, self.mixture_depth,
                self.aug_severity)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return self._aug(x), y
        else:
            im_tuple = (
                self.transform(x),
                self._aug(x),
                self._aug(x),
            )
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

