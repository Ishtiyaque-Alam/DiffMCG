"""
DiffMCG Diffusion Pipeline.

Three-stream diffusion process:
  - y_0 (ground-truth label) → y_t
  - ŷ_i (image prediction)   → y_t^i
  - ŷ_m (mask prediction)    → y_t^m

Each stream is noised independently with the same schedule but different noise samples.

Scheduler: DDPM/DDIM with linear beta schedule.
"""

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler


# ============================================================================
# DiffMCG Scheduler — three-stream noise addition
# ============================================================================

class DiffMCGScheduler(DDIMScheduler):
    """
    DDIM scheduler extended for DiffMCG's three-stream diffusion.
    """
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

    def add_noise_single(self, original, noise, timesteps):
        """
        Standard forward diffusion: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

        Args:
            original: (B, C) clean signal
            noise: (B, C) Gaussian noise
            timesteps: (B,) timestep indices
        Returns:
            noisy: (B, C) noisy signal
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original.device, dtype=original.dtype
        )
        timesteps = timesteps.to(original.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy

    def add_noise_three_stream(self, y_0, y_hat_i, y_hat_m, timesteps):
        """
        Add noise independently to all three streams.

        Args:
            y_0: (B, C) ground-truth label (one-hot or prototype)
            y_hat_i: (B, C) image encoder prediction
            y_hat_m: (B, C) mask encoder prediction
            timesteps: (B,) timestep indices
        Returns:
            y_t: (B, C) noisy ground truth
            y_t_i: (B, C) noisy image prediction
            y_t_m: (B, C) noisy mask prediction
            noise_y: (B, C) noise added to y_0 (target for denoiser)
            noise_i: (B, C) noise added to y_hat_i
            noise_m: (B, C) noise added to y_hat_m
        """
        noise_y = torch.randn_like(y_0)
        noise_i = torch.randn_like(y_hat_i)
        noise_m = torch.randn_like(y_hat_m)

        y_t = self.add_noise_single(y_0, noise_y, timesteps)
        y_t_i = self.add_noise_single(y_hat_i, noise_i, timesteps)
        y_t_m = self.add_noise_single(y_hat_m, noise_m, timesteps)

        return y_t, y_t_i, y_t_m, noise_y, noise_i, noise_m


# ============================================================================
# DiffMCG Sampler — reverse diffusion
# ============================================================================

class DiffMCGSampler:
    """
    Reverse diffusion sampler for DiffMCG.
    At each timestep, constructs the 5C input and calls the denoiser.
    """
    def __init__(self, model, scheduler, eta=0.0):
        self.model = model
        self.scheduler = scheduler
        self.eta = eta

    @torch.no_grad()
    def sample(self, image, mask):
        """
        Run reverse diffusion to predict clean label distribution.

        Args:
            image: (B, 3, H, W) input image
            mask: (B, 1, H, W) segmentation mask
        Returns:
            y_pred: (B, C) predicted label distribution
        """
        device = next(self.model.parameters()).device
        image = image.to(device)
        mask = mask.to(device)

        self.model.eval()

        # Get MCG features (clean, no noise)
        y_hat_i, y_hat_m = self.model.forward_mcg_only(image, mask)
        B, C = y_hat_i.shape

        # Initialize with random noise for all three streams
        y_t = torch.randn(B, C, device=device)
        y_t_i = torch.randn(B, C, device=device)
        y_t_m = torch.randn(B, C, device=device)

        # Reverse diffusion loop
        for t in self.scheduler.timesteps:
            timesteps = t * torch.ones(B, dtype=torch.long, device=device)

            # Construct 5C input
            denoiser_input = torch.cat([y_t_i, y_hat_i, y_t_m, y_hat_m, y_t], dim=1)

            # Predict noise
            noise_pred = self.model.denoiser(denoiser_input, timesteps)

            # DDIM step on y_t (the label stream — we want to recover y_0)
            y_t = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=y_t,
            ).prev_sample

            del noise_pred
            torch.cuda.empty_cache()

        return y_t


# ============================================================================
# Factory functions
# ============================================================================

def create_scheduler(opt, phase):
    """Create a DiffMCG scheduler based on config."""
    steps = opt['num_train_timesteps'] if phase == "train" else opt['num_test_timesteps']
    scheduler = DiffMCGScheduler(
        num_train_timesteps=steps,
        beta_start=opt['beta_start'],
        beta_end=opt['beta_end'],
        beta_schedule=opt['beta_schedule'],
    )
    return scheduler


def create_sampler(model, opt):
    """Create a DiffMCG sampler for inference."""
    scheduler = create_scheduler(opt, "test")
    scheduler.set_timesteps(opt['num_test_timesteps'])
    sampler = DiffMCGSampler(
        model=model,
        scheduler=scheduler,
        eta=opt.get('eta', 0.0),
    )
    return sampler
