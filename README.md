# DiffMCG

**Diffusion-based Medical Image Classification with Mask-Conditioned Guidance**

Implementation of the DiffMCG paper for medical image classification using diffusion models with mask-conditioned guidance.

## Architecture

DiffMCG consists of three main components:

1. **MCG Module**: Two parallel ResNet50 encoders (image + mask) producing C-dimensional class logits
2. **MLP Denoiser**: Time-conditioned MLP taking 5C-dimensional input `[y_t^i, ŷ_i, y_t^m, ŷ_m, y_t]`
3. **Three-stream diffusion**: Independent noise addition to ground-truth labels, image predictions, and mask predictions

### Loss Function
```
L_total = L_ε (diffusion MSE) + L_MCG (cross-entropy) + λ · L_MMD (three-way MMD)
```

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Dataset
Prepare ISIC skin lesion dataset with segmentation masks:
- Create pickle files with format: `[{'img_root': path, 'label': int, 'mask_root': path}, ...]`
- Update paths in `configs/isic.yml`

## Training

### Full two-stage training
```bash
python diffuser_trainer.py
```

This runs:
1. **Stage 1**: MCG pretraining (50 epochs)
2. **Stage 2**: End-to-end DiffMCG training (200 epochs)

### Monitor training
```bash
tensorboard --logdir logs/
```

## Project Structure
```
DiffMCG/
├── configs/isic.yml           # Dataset & model config
├── option/diff_DDIM.yaml      # Diffusion scheduler config
├── dataloader/
│   ├── loading.py             # ISIC dataset with mask support
│   └── transforms.py          # Joint image-mask augmentations
├── model.py                   # MCG module + MLP denoiser
├── pipeline.py                # Three-stream diffusion scheduler
├── diffuser_trainer.py        # Two-stage training pipeline
├── utils.py                   # MMD loss, metrics, helpers
├── optimizer.py               # Lion + SAM optimizers
└── requirements.txt
```

## Citation
```
@article{diffmcg2026,
  title={DiffMCG: Diffusion-based Medical Image Classification with Mask-Conditioned Guidance},
  journal={Neural Networks},
  year={2026}
}
```
