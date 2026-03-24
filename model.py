"""
DiffMCG Model Architecture.

Components:
  1. MCGModule — Mask-Conditioned Guiding with dual ResNet50 encoders
  2. ConditionalLinear — Timestep-conditioned linear layer
  3. MLPDenoiser — MLP-based denoiser with residual connections
  4. DiffMCG — Full model combining MCG + diffusion denoising

Architecture (from paper):
  - Two parallel ResNet50 (image & mask) → 1×1 conv → AdaptiveAvgPool → C-dim logits
  - MLP denoiser takes 5C input: [y_t^i, ŷ_i, y_t^m, ŷ_m, y_t] → predicts noise ε
  - Denoiser uses Linear→BN→Softplus with Hadamard product timestep embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


# ============================================================================
# Timestep-conditioned layers
# ============================================================================

class ConditionalLinear(nn.Module):
    """
    Linear layer conditioned on diffusion timestep via learned embedding.
    Output = embedding(t) ⊙ Linear(x)   (Hadamard product)
    """
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out  # Hadamard product
        return out


# ============================================================================
# MCG Module — Mask-Conditioned Guiding
# ============================================================================

class ResNet50Encoder(nn.Module):
    """
    ResNet50 feature extractor that produces C-dimensional class logits.
    Architecture: ResNet50 (no FC) → 1×1 Conv → AdaptiveAvgPool → Flatten
    """
    def __init__(self, num_classes, input_channels=3, pretrained=True):
        super(ResNet50Encoder, self).__init__()

        # Load ResNet50 backbone
        backbone = resnet50(pretrained=pretrained)
        self.featdim = backbone.fc.in_features  # 2048

        # Remove FC layer, keep everything else
        layers = []
        for name, module in backbone.named_children():
            if name != 'fc':
                layers.append(module)
        self.backbone = nn.Sequential(*layers)

        # If input is not 3-channel (e.g., 1-channel mask), replace first conv
        if input_channels != 3:
            self.backbone[0] = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 1×1 conv to map features to num_classes channels
        self.channel_conv = nn.Conv2d(self.featdim, num_classes, kernel_size=1, stride=1)
        # Adaptive pooling to 1×1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W) input image or mask
        Returns:
            logits: (B, num_classes) class logits
            features: (B, feat_dim) intermediate features (before 1×1 conv)
        """
        feat = self.backbone(x)             # (B, 2048, 7, 7) for 224×224 input
        logits_map = self.channel_conv(feat) # (B, num_classes, 7, 7)
        logits = self.pool(logits_map)       # (B, num_classes, 1, 1)
        logits = torch.flatten(logits, 1)    # (B, num_classes)
        return logits


class MCGModule(nn.Module):
    """
    Mask-Conditioned Guiding Module.
    Two parallel ResNet50 encoders:
      - Image encoder: RGB image → ŷ_i (C-dim logits)
      - Mask encoder: Segmentation mask → ŷ_m (C-dim logits)
    """
    def __init__(self, num_classes, pretrained=True):
        super(MCGModule, self).__init__()
        self.image_encoder = ResNet50Encoder(
            num_classes=num_classes,
            input_channels=3,
            pretrained=pretrained,
        )
        self.mask_encoder = ResNet50Encoder(
            num_classes=num_classes,
            input_channels=1,  # Binary mask
            pretrained=pretrained,
        )

    def forward(self, image, mask):
        """
        Args:
            image: (B, 3, H, W) RGB medical image
            mask: (B, 1, H, W) segmentation mask
        Returns:
            y_hat_i: (B, C) image branch logits
            y_hat_m: (B, C) mask branch logits
        """
        y_hat_i = self.image_encoder(image)
        y_hat_m = self.mask_encoder(mask)
        return y_hat_i, y_hat_m


# ============================================================================
# MLP Denoiser
# ============================================================================

class MLPDenoiser(nn.Module):
    """
    Time-conditioned MLP denoiser for DiffMCG.

    Input: 5C-dimensional concatenation [y_t^i, ŷ_i, y_t^m, ŷ_m, y_t]
    Output: C-dimensional predicted noise ε

    Architecture (from paper Fig. 3):
      ConditionalLinear(5C → hidden) → BN → Softplus
      → ConditionalLinear(hidden → hidden) → BN → Softplus
      → ConditionalLinear(hidden → hidden) → BN → Softplus
      → Linear(hidden → C)
      With residual connection from layer 1 output to pre-final layer
    """
    def __init__(self, num_classes, hidden_dim, n_steps):
        super(MLPDenoiser, self).__init__()
        input_dim = 5 * num_classes  # [y_t^i, ŷ_i, y_t^m, ŷ_m, y_t]

        # Layer 1
        self.lin1 = ConditionalLinear(input_dim, hidden_dim, n_steps)
        self.norm1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        # Layer 3
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.norm3 = nn.BatchNorm1d(hidden_dim)

        # Output projection
        self.lin4 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, t):
        """
        Args:
            x: (B, 5C) concatenated input [y_t^i, ŷ_i, y_t^m, ŷ_m, y_t]
            t: (B,) integer timestep indices
        Returns:
            noise_pred: (B, C) predicted noise
        """
        # Layer 1
        h = self.lin1(x, t)
        h = self.norm1(h)
        h = F.softplus(h)
        skip = h  # Residual connection

        # Layer 2
        h = self.lin2(h, t)
        h = self.norm2(h)
        h = F.softplus(h)

        # Layer 3
        h = self.lin3(h, t)
        h = self.norm3(h)
        h = F.softplus(h)

        # Residual connection
        h = h + skip

        # Output
        noise_pred = self.lin4(h)
        return noise_pred


# ============================================================================
# Full DiffMCG Model
# ============================================================================

class DiffMCG(nn.Module):
    """
    Complete DiffMCG model.
    Combines MCG module (dual ResNet50 encoders) with MLP denoiser.
    """
    def __init__(self, config):
        super(DiffMCG, self).__init__()
        self.num_classes = config.data.num_classes
        n_steps = config.diffusion.timesteps + 1
        hidden_dim = config.model.hidden_dim

        # MCG Module — dual ResNet50 encoders
        self.mcg = MCGModule(
            num_classes=self.num_classes,
            pretrained=True,
        )

        # MLP Denoiser
        self.denoiser = MLPDenoiser(
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            n_steps=n_steps,
        )

    def forward(self, image, mask, y_t, y_t_i, y_t_m, timesteps):
        """
        Full forward pass.

        Args:
            image: (B, 3, H, W) input medical image
            mask: (B, 1, H, W) segmentation mask
            y_t: (B, C) noisy ground-truth label
            y_t_i: (B, C) noisy image prediction
            y_t_m: (B, C) noisy mask prediction
            timesteps: (B,) diffusion timestep indices

        Returns:
            noise_pred: (B, C) predicted noise
            y_hat_i: (B, C) image encoder logits (for MCG loss)
            y_hat_m: (B, C) mask encoder logits (for MCG loss)
        """
        # MCG: extract features from image and mask
        y_hat_i, y_hat_m = self.mcg(image, mask)

        # Concatenate 5C input for denoiser: [y_t^i, ŷ_i, y_t^m, ŷ_m, y_t]
        denoiser_input = torch.cat([y_t_i, y_hat_i, y_t_m, y_hat_m, y_t], dim=1)

        # Denoise
        noise_pred = self.denoiser(denoiser_input, timesteps)

        return noise_pred, y_hat_i, y_hat_m

    def forward_mcg_only(self, image, mask):
        """MCG-only forward pass for Stage 1 pretraining."""
        y_hat_i, y_hat_m = self.mcg(image, mask)
        return y_hat_i, y_hat_m
