"""
Custom transforms for DiffMCG.
Supports joint image-mask geometric transforms to ensure consistent augmentation.
"""

from __future__ import division

import numbers
import random

import cv2
import numpy as np
from PIL import Image, ImageOps


class CropCenterSquare(object):
    """Crop the center square from a PIL image."""
    def __call__(self, img):
        img_w, img_h = img.size
        h = min(img_h, img_w)
        crop = CenterCrop(h)
        return crop(img)


class CropCenterSquareJoint(object):
    """Crop center square from image and mask jointly."""
    def __call__(self, img, mask):
        img_w, img_h = img.size
        h = min(img_h, img_w)
        crop = CenterCropJoint(h)
        return crop(img, mask)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        return img


class CenterCropJoint(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        return img, mask


class RandomRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR)


class RandomRotationJoint(object):
    """Apply the same random rotation to image and mask."""
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            img.rotate(rotate_degree, Image.BILINEAR),
            mask.rotate(rotate_degree, Image.NEAREST),
        )


class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomHorizontalFlipJoint(object):
    """Apply the same random horizontal flip to image and mask."""
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomVerticalFlipJoint(object):
    """Apply the same random vertical flip to image and mask."""
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class adjust_light(object):
    def __call__(self, image):
        seed = random.random()
        if seed > 0.5:
            gamma = random.random() * 3 + 0.5
            invGamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
            ).astype(np.uint8)
            image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)
        return image


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        w, h = img.size
        ch, cw = self.size
        if w == cw and h == ch:
            return img
        if w < cw or h < ch:
            pw = cw - w if cw > w else 0
            ph = ch - h if ch > h else 0
            padding = (pw, ph, pw, ph)
            img = ImageOps.expand(img, padding, fill=0)
            w, h = img.size
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        return img.crop((x1, y1, x1 + cw, y1 + ch))
