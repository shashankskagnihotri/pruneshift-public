# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as trn

IMAGE_SIZE = None

def set_image_size(size):
    global IMAGE_SIZE
    IMAGE_SIZE = size


convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
#def contrast(pil_img, level):
#    level = float_parameter(sample_level(level), 1.8) + 0.1
#    return ImageEnhance.Contrast(pil_img).enhance(level)

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    out = np.clip((x - means) * c + means, 0, 1) * 255
    out = np.uint8(out)
    return convert_img(out)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


class Cutout(object):
    def __init__(self, n_holes, max_height, max_width, min_height=None, min_width=None,
                 fill_value_mode='rand', p=0.5):
        self.n_holes = n_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_width = min_width if min_width is not None else max_width
        self.min_height = min_height if min_height is not None else max_height
        self.fill_value_mode = fill_value_mode  # 'zero' 'one' 'uniform'
        self.p = p
        assert 0 < self.min_height <= self.max_height
        assert 0 < self.min_width <= self.max_width
        assert 0 < self.n_holes
        #assert self.fill_value_mode in ['zero', 'one', 'uniform']

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]

        if self.fill_value_mode == 'rand':
            modes = ['uniform', 'gauss']
            self.fill_value_mode = np.random.choice(modes)

        if self.fill_value_mode == 'uniform':
            f = lambda : img*np.random.uniform(**{'low': 0.1, 'high': 2, 'size': (h, w, 3)})
        elif self.fill_value_mode == 'gauss':
            f = lambda: img*np.random.normal(loc=0.5, scale=1.5, size=(h,w,3))
        else:
            raise("Exception")

        mask = np.ones((h, w, 3), dtype=np.int32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            h_l = np.random.randint(self.min_height, self.max_height + 1)
            w_l = np.random.randint(self.min_width, self.max_width + 1)

            y1 = np.clip(y - h_l // 2, 0, h)
            y2 = np.clip(y + h_l // 2, 0, h)
            x1 = np.clip(x - w_l // 2, 0, w)
            x2 = np.clip(x + w_l // 2, 0, w)
            mask[y1:y2, x1:x2, :] = 0
        img = np.where(mask, img, f())
        return img, mask

def fft_aug(pil_img, severity=None):
    img_np = np.array(pil_img)
    fft = np.fft.fft2(img_np, axes = [0,1])
    holes = np.random.randint(2, 8)
    co = Cutout(holes, 64, 64, 48, 48)
    fft_cutout, mask = co(fft)
    ifft = np.fft.ifft2(fft_cutout, axes = [0,1])
    real_img = np.real(ifft)
    real_img = np.clip(real_img, 0, 255).astype(np.uint8)
    aug_img = convert_img(real_img)
    return aug_img


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y,
]

augmentations_test = [ brightness, contrast, sharpness, color]


augmentations_all = augmentations + augmentations_test
