"""
Utility functions for DiffMCG.
Includes:
  - Optimizer factory
  - Dataset factory
  - Label conversion
  - Metrics (accuracy, F1, AUC, kappa, etc.)
  - MMD regularization loss (Gaussian kernel)
  - KL divergence utilities
"""

import random
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score,
)


# ============================================================================
# Random seed
# ============================================================================

def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# Config helpers
# ============================================================================

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# ============================================================================
# Optimizer factory
# ============================================================================

def get_optimizer(config_optim, parameters):
    if config_optim.optimizer == 'Adam':
        return optim.Adam(
            parameters, lr=config_optim.lr,
            weight_decay=config_optim.weight_decay,
            betas=(config_optim.beta1, 0.999),
            amsgrad=config_optim.amsgrad,
            eps=config_optim.eps,
        )
    elif config_optim.optimizer == 'AdamW':
        return optim.AdamW(
            parameters, lr=config_optim.lr,
            weight_decay=0.05,
            betas=(config_optim.beta1, 0.999),
            eps=1e-8,
        )
    elif config_optim.optimizer == 'SGD':
        return optim.SGD(
            parameters, lr=config_optim.lr,
            weight_decay=1e-4, momentum=0.9,
        )
    else:
        raise NotImplementedError(f'Optimizer {config_optim.optimizer} not understood.')


# ============================================================================
# Dataset factory
# ============================================================================

def get_dataset(config):
    from dataloader.loading import ISIC2017Dataset

    data_object = None

    if config.data.dataset == "ISIC2017":
        train_dataset = ISIC2017Dataset(
            image_dir=config.data.train_image_dir,
            mask_dir=config.data.train_mask_dir,
            label_csv=config.data.train_label_csv,
            train=True,
        )
        test_dataset = ISIC2017Dataset(
            image_dir=config.data.val_image_dir,
            mask_dir=config.data.val_mask_dir,
            label_csv=config.data.val_label_csv,
            train=False,
        )
    else:
        raise NotImplementedError(
            f"Dataset {config.data.dataset} not implemented. Only ISIC2017 is supported."
        )
    return data_object, train_dataset, test_dataset


# ============================================================================
# Label conversion
# ============================================================================

def cast_label_to_one_hot_and_prototype(y_labels_batch, config, return_prototype=True):
    """
    Convert integer labels to one-hot and prototype logits.
    y_labels_batch: (batch_size,) integer tensor
    Returns: (one_hot, logits) tensors of shape (batch_size, num_classes)
    """
    y_one_hot_batch = F.one_hot(
        y_labels_batch, num_classes=config.data.num_classes
    ).float()

    if return_prototype:
        label_min, label_max = config.data.label_min_max
        y_logits_batch = torch.logit(
            F.normalize(
                torch.clip(y_one_hot_batch, min=label_min, max=label_max),
                p=1.0, dim=1,
            )
        )
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch


# ============================================================================
# MMD Regularization Loss (Gaussian kernel)
# ============================================================================

def compute_kernel(x, y):
    """Gaussian kernel between x and y."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    """Maximum Mean Discrepancy between distributions x and y."""
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def mmd_three_way_loss(y_img, y_mask, y_label):
    """
    Three-way MMD regularization from DiffMCG paper.
    Enforces distributional consistency between:
      1. Image prediction ↔ Ground-truth label
      2. Mask prediction ↔ Ground-truth label
      3. Image prediction ↔ Mask prediction

    Args:
        y_img: (batch_size, num_classes) image branch logits
        y_mask: (batch_size, num_classes) mask branch logits
        y_label: (batch_size, num_classes) one-hot ground truth
    Returns:
        scalar MMD loss
    """
    mmd_img_label = compute_mmd(y_img, y_label)
    mmd_mask_label = compute_mmd(y_mask, y_label)
    mmd_img_mask = compute_mmd(y_img, y_mask)
    return mmd_img_label + mmd_mask_label + mmd_img_mask


# ============================================================================
# Classification metrics
# ============================================================================

def compute_isic_metrics(gt, pred):
    """Compute classification metrics for ISIC-style tasks."""
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()

    gt_class = np.argmax(gt_np, axis=1)
    pred_class = np.argmax(pred_np, axis=1)

    ACC = accuracy_score(gt_class, pred_class)
    BACC = balanced_accuracy_score(gt_class, pred_class)
    Prec = precision_score(gt_class, pred_class, average='macro', zero_division=0)
    Rec = recall_score(gt_class, pred_class, average='macro', zero_division=0)
    F1 = f1_score(gt_class, pred_class, average='macro', zero_division=0)

    try:
        AUC_ovo = roc_auc_score(gt_np, pred_np, multi_class='ovo', average='macro')
    except Exception:
        AUC_ovo = 0.0

    kappa = cohen_kappa_score(gt_class, pred_class, weights='quadratic')
    return ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa


def compute_f1_score(gt, pred):
    """Compute macro F1 score."""
    gt_class = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    pred_class = np.argmax(pred_np, axis=1)
    F1 = f1_score(gt_class, pred_class, average='macro', zero_division=0)
    return F1


# ============================================================================
# KL divergence utilities
# ============================================================================

def categorical_kl_logits(logits1, logits2, eps=1.e-6):
    """KL(C(logits1) || C(logits2))"""
    out = (
        F.softmax(logits1 + eps, dim=-1)
        * (F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1))
    )
    return torch.sum(out, dim=-1)


def meanflat(x):
    """Mean over all axes except first batch dimension."""
    return x.mean(dim=tuple(range(1, len(x.shape))))
