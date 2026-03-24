"""
DiffMCG Training Pipeline.

Two-stage training:
  Stage 1: Pretrain MCG module only (50 epochs, Adam, lr=2e-4)
           Loss = CE(ŷ_i, y₀) + CE(ŷ_m, y₀)
  Stage 2: End-to-end training (200 epochs, Adam, lr=1e-3)
           Loss = L_ε + L_MCG + λ·L_MMD

Uses PyTorch Lightning for training orchestration.
"""

from typing import Optional
import os
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

import pipeline
from model import DiffMCG
from utils import (
    get_optimizer, get_dataset,
    cast_label_to_one_hot_and_prototype,
    compute_isic_metrics, mmd_three_way_loss,
)


# ============================================================================
# Stage 1: MCG Pretraining Module
# ============================================================================

class MCGPretrainer(pl.LightningModule):
    """
    Stage 1: Pretrain MCG module only.
    Loss = CE(ŷ_i, y₀) + CE(ŷ_m, y₀)
    """

    def __init__(self, hparams):
        super(MCGPretrainer, self).__init__()
        self.params = hparams
        self.epochs = self.params.mcg.pretrain_epochs
        self.initlr = self.params.mcg_optim.lr

        self.model = DiffMCG(self.params)
        self.ce_loss = nn.CrossEntropyLoss()

        self.save_hyperparameters()
        self.gts = []
        self.preds = []

    def configure_optimizers(self):
        # Only optimize MCG parameters
        mcg_params = list(self.model.mcg.parameters())
        optimizer = torch.optim.Adam(
            mcg_params,
            lr=self.initlr,
            betas=(self.params.mcg_optim.beta1, 0.999),
            eps=self.params.mcg_optim.eps,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x_batch, y_batch, mask_batch = batch
        x_batch = x_batch.cuda()
        mask_batch = mask_batch.cuda()

        # One-hot labels for CE loss target
        y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_labels = y_batch.cuda()

        # MCG forward
        y_hat_i, y_hat_m = self.model.forward_mcg_only(x_batch, mask_batch)

        # MCG loss: CE(ŷ_i, y₀) + CE(ŷ_m, y₀)
        loss_img = self.ce_loss(y_hat_i, y_labels)
        loss_mask = self.ce_loss(y_hat_m, y_labels)
        loss = loss_img + loss_mask

        self.log("mcg_train_loss", loss, prog_bar=True)
        self.log("mcg_loss_img", loss_img)
        self.log("mcg_loss_mask", loss_mask)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.gts) == 0:
            return
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

        self.log('mcg_accuracy', ACC)
        self.log('mcg_f1', F1)
        self.log('mcg_auc', AUC_ovo)

        self.gts = []
        self.preds = []
        print(f"MCG Pretrain Val: Acc={ACC:.4f}, F1={F1:.4f}, AUC={AUC_ovo:.4f}")

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch, mask_batch = batch
        y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        x_batch = x_batch.cuda()
        mask_batch = mask_batch.cuda()

        y_hat_i, y_hat_m = self.model.forward_mcg_only(x_batch, mask_batch)
        # Average image and mask predictions
        y_pred = (y_hat_i + y_hat_m) / 2.0

        self.preds.append(y_pred)
        self.gts.append(y_one_hot.cuda())

    def train_dataloader(self):
        _, train_dataset, _ = get_dataset(self.params)
        return DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=self.params.data.num_workers,
        )

    def val_dataloader(self):
        _, _, test_dataset = get_dataset(self.params)
        return DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
        )


# ============================================================================
# Stage 2: End-to-End Training Module
# ============================================================================

class DiffMCGTrainer(pl.LightningModule):
    """
    Stage 2: End-to-end DiffMCG training.
    Loss = L_ε + L_MCG + λ·L_MMD

    L_ε   = MSE(ε_pred, ε)         (diffusion noise prediction)
    L_MCG = CE(ŷ_i, y₀) + CE(ŷ_m, y₀)  (MCG supervision)
    L_MMD = MMD(ŷ_i, y₀) + MMD(ŷ_m, y₀) + MMD(ŷ_i, ŷ_m)  (distributional consistency)
    """

    def __init__(self, hparams, pretrained_mcg_ckpt=None):
        super(DiffMCGTrainer, self).__init__()
        self.params = hparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr
        self.lambda_mmd = self.params.training.lambda_mmd

        # Load scheduler config
        config_path = r'/kaggle/working/DiffMCG/option/diff_DDIM.yaml'
        with open(config_path, 'r') as f:
            diff_params = yaml.safe_load(f)
        self.diff_opt = EasyDict(diff_params)

        # Model
        self.model = DiffMCG(self.params)

        # Load pretrained MCG weights if provided
        if pretrained_mcg_ckpt is not None:
            self._load_mcg_weights(pretrained_mcg_ckpt)

        self.ce_loss = nn.CrossEntropyLoss()

        self.save_hyperparameters()
        self.gts = []
        self.preds = []

        # Scheduler for diffusion
        self.train_scheduler = pipeline.create_scheduler(
            self.diff_opt['scheduler'], 'train'
        )

        # Sampler for inference
        self.sampler = pipeline.create_sampler(
            self.model, self.diff_opt['scheduler']
        )

    def _load_mcg_weights(self, ckpt_path):
        """Load pretrained MCG weights from Stage 1."""
        print(f"Loading pretrained MCG weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter MCG-related keys
        mcg_state = {}
        for k, v in state_dict.items():
            # Handle Lightning prefix
            clean_key = k.replace('model.', '', 1) if k.startswith('model.') else k
            if clean_key.startswith('mcg.'):
                mcg_state[clean_key] = v

        if mcg_state:
            model_state = self.model.state_dict()
            model_state.update(mcg_state)
            self.model.load_state_dict(model_state)
            print(f"Loaded {len(mcg_state)} MCG parameters")
        else:
            print("WARNING: No MCG parameters found in checkpoint")

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.params.optim,
            filter(lambda p: p.requires_grad, self.model.parameters()),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.model.train()
        x_batch, y_batch, mask_batch = batch

        # Convert labels
        y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_one_hot = y_one_hot.cuda()
        y_labels = y_batch.cuda()
        x_batch = x_batch.cuda()
        mask_batch = mask_batch.cuda()

        B = x_batch.shape[0]
        C = self.params.data.num_classes

        # --- MCG forward (get ŷ_i and ŷ_m) ---
        y_hat_i, y_hat_m = self.model.forward_mcg_only(x_batch, mask_batch)

        # --- Forward diffusion: add noise to all three streams ---
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()

        y_t, y_t_i, y_t_m, noise_y, noise_i, noise_m = \
            self.train_scheduler.add_noise_three_stream(
                y_one_hot, y_hat_i.detach(), y_hat_m.detach(), timesteps
            )

        # --- Denoiser forward ---
        denoiser_input = torch.cat([y_t_i, y_hat_i, y_t_m, y_hat_m, y_t], dim=1)
        noise_pred = self.model.denoiser(denoiser_input, timesteps)

        # --- Losses ---
        # L_ε: MSE between predicted and true noise (on y_0 stream)
        loss_diffusion = F.mse_loss(noise_pred, noise_y)

        # L_MCG: Cross-entropy supervision for both encoder branches
        loss_mcg = self.ce_loss(y_hat_i, y_labels) + self.ce_loss(y_hat_m, y_labels)

        # L_MMD: Three-way distributional consistency
        loss_mmd = mmd_three_way_loss(
            y_hat_i.softmax(1), y_hat_m.softmax(1), y_one_hot
        )

        # Total loss
        loss = loss_diffusion + loss_mcg + self.lambda_mmd * loss_mmd

        # Logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_diffusion", loss_diffusion)
        self.log("loss_mcg", loss_mcg)
        self.log("loss_mmd", loss_mmd)

        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.gts) == 0:
            return
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

        self.log('accuracy', ACC)
        self.log('f1', F1)
        self.log('Precision', Prec)
        self.log('Recall', Rec)
        self.log('AUC', AUC_ovo)
        self.log('kappa', kappa)

        self.gts = []
        self.preds = []
        print(
            f"Val: Accuracy={ACC:.4f}, F1={F1:.4f}, Precision={Prec:.4f}, "
            f"Recall={Rec:.4f}, AUC={AUC_ovo:.4f}, Kappa={kappa:.4f}"
        )

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x_batch, y_batch, mask_batch = batch

        y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_one_hot = y_one_hot.cuda()
        x_batch = x_batch.cuda()
        mask_batch = mask_batch.cuda()

        # Run reverse diffusion to get predicted label
        y_pred = self.sampler.sample(x_batch, mask_batch)

        self.preds.append(y_pred)
        self.gts.append(y_one_hot)

    def train_dataloader(self):
        _, train_dataset, _ = get_dataset(self.params)
        return DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=self.params.data.num_workers,
        )

    def val_dataloader(self):
        _, _, test_dataset = get_dataset(self.params)
        return DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
        )


# ============================================================================
# Main entry point
# ============================================================================

def main():
    # ---- Configuration ----
    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    config_path = r'/kaggle/working/DiffMCG/configs/isic.yml'
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    output_dir = 'logs'
    version_name = 'DiffMCG'

    # ================================================================
    # Stage 1: Pretrain MCG Module
    # ================================================================
    print("=" * 60)
    print("Stage 1: Pretraining MCG Module")
    print("=" * 60)

    mcg_logger = TensorBoardLogger(
        name='diffmcg_mcg_pretrain', save_dir=output_dir
    )

    mcg_model = MCGPretrainer(config)

    mcg_checkpoint = ModelCheckpoint(
        monitor='mcg_f1',
        filename='mcg-epoch{epoch:02d}-f1-{mcg_f1:.4f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
        save_last=True,
    )
    mcg_lr_monitor = LearningRateMonitor(logging_interval='step')

    mcg_trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=config.mcg.pretrain_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=32,
        logger=mcg_logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=5,
        callbacks=[mcg_checkpoint, mcg_lr_monitor],
    )

    mcg_trainer.fit(mcg_model)
    mcg_best_ckpt = mcg_checkpoint.best_model_path
    print(f"MCG pretraining complete. Best checkpoint: {mcg_best_ckpt}")

    # ================================================================
    # Stage 2: End-to-End Training
    # ================================================================
    print("=" * 60)
    print("Stage 2: End-to-End DiffMCG Training")
    print("=" * 60)

    e2e_logger = TensorBoardLogger(
        name='diffmcg_e2e', save_dir=output_dir
    )

    e2e_model = DiffMCGTrainer(config, pretrained_mcg_ckpt=mcg_best_ckpt)

    e2e_checkpoint = ModelCheckpoint(
        monitor='f1',
        filename='diffmcg-epoch{epoch:02d}-acc-{accuracy:.4f}-f1-{f1:.4f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
        save_last=True,
    )
    e2e_lr_monitor = LearningRateMonitor(logging_interval='step')

    e2e_trainer = pl.Trainer(
        check_val_every_n_epoch=config.training.validation_freq,
        max_epochs=config.training.n_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=32,
        logger=e2e_logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=5,
        callbacks=[e2e_checkpoint, e2e_lr_monitor],
    )

    e2e_trainer.fit(e2e_model)

    # Optional: Validate best model
    # e2e_trainer.validate(e2e_model, ckpt_path=e2e_checkpoint.best_model_path)


if __name__ == '__main__':
    main()
