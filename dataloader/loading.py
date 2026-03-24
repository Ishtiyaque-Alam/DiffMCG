"""
Dataset classes for DiffMCG — ISIC 2017 Kaggle dataset.

ISIC 2017 structure:
  training_images/   (2000 JPEG images)
  training_masks/    (2000 PNG binary masks, suffix _segmentation)
  val_images/        (150 images)
  val_masks/         (150 masks)
  test_images/       (600 images)
  test_masks/        (600 masks)
  training_labels.csv / val_labels.csv / test_labels.csv

Labels CSV format:
  image_id, melanoma, seborrheic_keratosis
  - If melanoma=1: class 0
  - If seborrheic_keratosis=1: class 1
  - Otherwise (nevus): class 2
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import dataloader.transforms as trans

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ISIC2017Dataset(Dataset):
    """
    ISIC 2017 Skin Lesion Dataset with segmentation mask support.

    Loads images from image_dir, paired masks from mask_dir,
    and labels from a CSV file.
    """

    def __init__(self, image_dir, mask_dir, label_csv, train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.trainsize = (224, 224)

        # Load labels from CSV
        self.labels_df = pd.read_csv(label_csv)
        self.image_ids = self.labels_df.iloc[:, 0].values  # First column = image_id

        # Parse labels: melanoma=0, seborrheic_keratosis=1, nevus=2
        self.labels = self._parse_labels()

        # Discover image files and build mapping
        self.image_paths, self.mask_paths = self._discover_files()
        self.size = len(self.image_paths)
        print(f"ISIC2017Dataset: found {self.size} samples (train={train})")

        # Joint geometric transforms
        self.joint_flip = trans.RandomHorizontalFlipJoint()
        self.joint_rotation = trans.RandomRotationJoint(30)
        self.joint_crop = trans.CropCenterSquareJoint()

        # Image-only normalization (ImageNet stats)
        self.img_normalize = transforms.Compose([
            transforms.Resize(self.trainsize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Mask-only resize + to tensor
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                self.trainsize,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def _parse_labels(self):
        """Parse CSV to get integer class labels."""
        labels = {}
        for _, row in self.labels_df.iterrows():
            img_id = str(row.iloc[0])
            melanoma = float(row.iloc[1])
            seborrheic = float(row.iloc[2])

            if melanoma >= 0.5:
                label = 0  # melanoma
            elif seborrheic >= 0.5:
                label = 1  # seborrheic_keratosis
            else:
                label = 2  # nevus
            labels[img_id] = label
        return labels

    def _discover_files(self):
        """Find all image files and their paired masks."""
        image_paths = []
        mask_paths = []

        for img_id in self.image_ids:
            img_id = str(img_id)

            # Find image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = os.path.join(self.image_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                # Try to find by glob
                matches = glob.glob(os.path.join(self.image_dir, img_id + '.*'))
                if matches:
                    img_path = matches[0]
                else:
                    continue  # Skip if image not found

            # Find mask file (ISIC 2017 uses _segmentation suffix)
            mask_path = None
            for suffix in ['_segmentation.png', '_segmentation.jpg', '_mask.png', '.png']:
                candidate = os.path.join(self.mask_dir, img_id + suffix)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                # Try glob
                matches = glob.glob(os.path.join(self.mask_dir, img_id + '*'))
                if matches:
                    mask_path = matches[0]
                else:
                    mask_path = None  # Will use dummy mask

            image_paths.append((img_id, img_path, mask_path))

        # Unpack
        ids = [x[0] for x in image_paths]
        imgs = [x[1] for x in image_paths]
        masks = [x[2] for x in image_paths]
        return list(zip(ids, imgs)), list(zip(ids, masks))

    def __getitem__(self, index):
        img_id, img_path = self.image_paths[index]
        _, mask_path = self.mask_paths[index]
        label = self.labels.get(str(img_id), 2)  # Default to nevus

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Load mask
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Dummy all-ones mask if not available
            mask = Image.new('L', img.size, 255)

        # Apply joint geometric transforms
        if self.train:
            img, mask = self.joint_crop(img, mask)
            img, mask = self.joint_flip(img, mask)
            img, mask = self.joint_rotation(img, mask)
        else:
            img, mask = self.joint_crop(img, mask)

        # Apply individual transforms
        img_torch = self.img_normalize(img)
        mask_torch = self.mask_transform(mask)
        mask_torch = (mask_torch > 0.5).float()  # Binarize

        return img_torch, label, mask_torch

    def __len__(self):
        return len(self.image_paths)
