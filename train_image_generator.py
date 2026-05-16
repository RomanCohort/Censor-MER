# =============================================================================
# Training Script: Biomimetic Image Generator
# =============================================================================
# Trains the BiomimeticImageGenerator model using:
#   - L2 reconstruction loss
#   - Perceptual loss (using VGG features)
#   - Illumination smoothness loss
#   - Sparse regularization
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import os
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

from model.biomimetic_image_generator import (
    BiomimeticImageGenerator,
    BiomimeticGenerationPipeline,
    AUToIllumination,
    create_biomimetic_generator
)
from visual_perception import VisualPerceptionPostProcess
from config.defaults import (
    VISUAL_PERCEPTION_CONFIG,
    AU_DECODER_CONFIG,
    DATA_CONFIG,
    FFA_CONFIG,
    AMYGDALA_CONFIG,
    CASA_CONFIG,
    SPARSE_CONTROL_CONFIG,
)


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingConfig:
    """Training hyperparameters"""

    def __init__(self):
        # Model architecture
        self.fast_dim = 512
        self.slow_dim = 768
        self.fused_dim = 1024

        # Training parameters
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.batch_size = 4
        self.num_workers = 2
        self.epochs = 50
        self.warmup_epochs = 5

        # Loss weights
        self.lambda_l2 = 1.0
        self.lambda_perceptual = 0.1
        self.lambda_smooth = 0.01
        self.lambda_sparse = 0.001
        self.lambda_contrastive = 0.05

        # Image parameters
        self.image_size = 224
        self.image_channels = 3

        # Checkpointing
        self.save_freq = 5
        self.eval_freq = 1
        self.max_keep = 5

        # Optimizer
        self.optimizer = 'AdamW'
        self.scheduler = 'CosineAnnealingWarmRestarts'
        self.scheduler_t0 = 10
        self.scheduler_t_mult = 2

        # Misc
        self.seed = 42
        self.gpu = 0
        self.use_amp = True  # Automatic mixed precision

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# =============================================================================
# Loss Functions
# =============================================================================

class L2Loss(nn.Module):
    """L2 reconstruction loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    Aligns high-level features between generated and target images.
    """

    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:17].eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg
        except Exception as e:
            print(f"Warning: Could not load VGG model: {e}")
            self.vgg = None

        self.layers = layers

    def forward(self, pred, target):
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)

        # Ensure 4D input
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # Resize to 224 if needed
        if pred.shape[-2:] != (224, 224):
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        return F.mse_loss(pred_features, target_features)


class IlluminationSmoothnessLoss(nn.Module):
    """
    Illumination smoothness loss.
    Penalizes sudden changes in illumination parameters over time.
    """

    def __init__(self):
        super().__init__()

    def forward(self, illum_params):
        """
        Args:
            illum_params: (B, T, 4) illumination parameters over time
        Returns:
            Loss value
        """
        if illum_params.dim() == 2:
            return torch.tensor(0.0, device=illum_params.device)

        # Compute temporal difference
        diff = illum_params[:, 1:] - illum_params[:, :-1]
        return (diff ** 2).mean()


class SparseRegularizationLoss(nn.Module):
    """
    Sparse regularization loss.
    Encourages sparse activations in the model.
    """

    def __init__(self, l1_factor=0.001):
        super().__init__()
        self.l1_factor = l1_factor

    def forward(self, model):
        l1_loss = 0.0
        for p in model.parameters():
            l1_loss += torch.abs(p).sum()
        return self.l1_factor * l1_loss


class ContrastiveAlignmentLoss(nn.Module):
    """
    Contrastive alignment loss for feature consistency.
    Ensures generated features align with target features in embedding space.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_pred, z_target):
        """
        Args:
            z_pred: (B, D) predicted features
            z_target: (B, D) target features
        Returns:
            Contrastive loss
        """
        # Normalize
        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target, dim=-1)

        # Similarity matrix
        sim = torch.matmul(z_pred, z_target.T) / self.temperature
        labels = torch.arange(len(z_pred), device=z_pred.device)

        return F.cross_entropy(sim, labels)


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR (Peak Signal-to-Noise Ratio)"""
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr.item()


def compute_ssim(pred, target, window_size=11):
    """
    Compute SSIM (Structural Similarity Index).
    Simplified implementation - for full version use pytorch-msssim.
    """
    # Convert to grayscale
    pred_gray = pred.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)

    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Local mean and variance
    mu_pred = F.avg_pool2d(pred_gray, window_size, 1, window_size // 2)
    mu_target = F.avg_pool2d(target_gray, window_size, 1, window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred_gray ** 2, window_size, 1, window_size // 2) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target_gray ** 2, window_size, 1, window_size // 2) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred_gray * target_gray, window_size, 1, window_size // 2) - mu_pred_target

    # SSIM
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
              ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


# =============================================================================
# Dataset Placeholder (Replace with actual data loading)
# =============================================================================

class ImageGenerationDataset(Dataset):
    """
    Placeholder dataset for image generation training.
    Replace this with actual data loading from your video dataset.
    """

    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.size = config.batch_size
        self.gen_data()

    def gen_data(self):
        """Generate dummy data for testing"""
        # Placeholder: Replace with actual data loading
        self.data = []
        for i in range(100):  # 100 samples
            self.data.append({
                'fast_feat': torch.randn(self.config.fast_dim),
                'slow_feat': torch.randn(self.config.slow_dim),
                'au_intensities': torch.rand(16, 28),
                'target_image': torch.rand(3, self.config.image_size, self.config.image_size),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'fast_feat': item['fast_feat'],
            'slow_feat': item['slow_feat'],
            'au_intensities': item['au_intensities'],
            'target': item['target_image'],
        }


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, dataloader, loss_fns, optimizer, device, config, epoch):
    """Train for one epoch"""

    model.train()
    total_loss = 0.0
    total_l2 = 0.0
    total_perceptual = 0.0
    total_smooth = 0.0
    total_sparse = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move to device
        fast_feat = batch['fast_feat'].to(device)
        slow_feat = batch['slow_feat'].to(device)
        au_intensities = batch['au_intensities'].to(device) if 'au_intensities' in batch else None
        target = batch['target'].to(device)

        optimizer.zero_grad()

        # Forward pass
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            generated = model(
                fast_feat=fast_feat,
                slow_feat=slow_feat,
                au_intensities=au_intensities,
                apply_visual_perception=True
            )

            # Compute losses
            loss_l2 = loss_fns['l2'](generated, target)
            loss_perceptual = loss_fns['perceptual'](generated, target)
            loss_smooth = loss_fns['smooth'](torch.rand(*au_intensities.shape[:2], 4).to(device) if au_intensities is not None else torch.zeros(1, 4).to(device))
            loss_sparse = loss_fns['sparse'](model)

            # Total loss
            loss = config.lambda_l2 * loss_l2 \
                 + config.lambda_perceptual * loss_perceptual \
                 + config.lambda_smooth * loss_smooth \
                 + config.lambda_sparse * loss_sparse

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_l2 += loss_l2.item()
        total_perceptual += loss_perceptual.item()
        total_smooth += loss_smooth.item()
        total_sparse += loss_sparse.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l2': f"{loss_l2.item():.4f}",
        })

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'l2': total_l2 / n,
        'perceptual': total_perceptual / n,
        'smooth': total_smooth / n,
        'sparse': total_sparse / n,
    }


@torch.no_grad()
def validate(model, dataloader, loss_fns, device, config):
    """Validation"""
    model.eval()

    total_l2 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for batch in dataloader:
        fast_feat = batch['fast_feat'].to(device)
        slow_feat = batch['slow_feat'].to(device)
        au_intensities = batch['au_intensities'].to(device) if 'au_intensities' in batch else None
        target = batch['target'].to(device)

        # Forward
        generated = model(
            fast_feat=fast_feat,
            slow_feat=slow_feat,
            au_intensities=au_intensities,
            apply_visual_perception=True
        )

        # Metrics
        l2 = F.mse_loss(generated, target)
        psnr = compute_psnr(generated, target)
        ssim = compute_ssim(generated, target)

        total_l2 += l2.item()
        total_psnr += psnr
        total_ssim += ssim
        n += 1

    return {
        'l2': total_l2 / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n,
    }


# =============================================================================
# Main Training Function
# =============================================================================

def train(config_path=None, output_dir='checkpoints'):
    """Main training function"""

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--output', type=str, default=output_dir)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Create config
    config = TrainingConfig()

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Create model
    print("Creating model...")
    model_config = {
        'fast_dim': config.fast_dim,
        'slow_dim': config.slow_dim,
        'fused_dim': config.fused_dim,
    }
    model = BiomimeticImageGenerator(model_config)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss functions
    loss_fns = {
        'l2': L2Loss(),
        'perceptual': PerceptualLoss(),
        'smooth': IlluminationSmoothnessLoss(),
        'sparse': SparseRegularizationLoss(config.lambda_sparse),
    }

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_t_mult
    )

    # Dataset
    train_dataset = ImageGenerationDataset(config, split='train')
    val_dataset = ImageGenerationDataset(config, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Training loop
    best_l2 = float('inf')
    history = []

    print(f"\nStarting training for {config.epochs} epochs...")

    for epoch in range(1, config.epochs + 1):
        # Warmup
        if epoch <= config.warmup_epochs:
            lr = config.lr * epoch / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Train
        train_metrics = train_one_epoch(model, train_loader, loss_fns, optimizer, device, config, epoch)

        # Validation
        if epoch % config.eval_freq == 0:
            val_metrics = validate(model, val_loader, loss_fns, device, config)
            print(f"\nEpoch {epoch}: train_l2={train_metrics['l2']:.4f}, val_l2={val_metrics['l2']:.4f}, "
                  f"psnr={val_metrics['psnr']:.2f}, ssim={val_metrics['ssim']:.3f}")
        else:
            print(f"\nEpoch {epoch}: train_l2={train_metrics['l2']:.4f}")
            val_metrics = {'l2': float('inf')}

        # Scheduler
        scheduler.step()

        # Save checkpoint
        if epoch % config.save_freq == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

            # Track best
            if val_metrics['l2'] < best_l2:
                best_l2 = val_metrics['l2']
                best_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, best_path)
                print(f"Saved best model: {best_path}")

        # History
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
        })

    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! History saved to {history_path}")
    return model


# =============================================================================
# Quick Training Test
# =============================================================================

def quick_test():
    """Quick training test with dummy data"""
    print("=" * 60)
    print(" Quick Training Test")
    print("=" * 60)

    config = TrainingConfig()
    config.batch_size = 2
    config.epochs = 2
    config.save_freq = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model_config = {
        'fast_dim': config.fast_dim,
        'slow_dim': config.slow_dim,
        'fused_dim': config.fused_dim,
    }
    model = BiomimeticImageGenerator(model_config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss functions
    loss_fns = {
        'l2': L2Loss(),
        'perceptual': PerceptualLoss(),
        'smooth': IlluminationSmoothnessLoss(),
        'sparse': SparseRegularizationLoss(config.lambda_sparse),
    }

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # Dummy data
    batch = {
        'fast_feat': torch.randn(config.batch_size, config.fast_dim).to(device),
        'slow_feat': torch.randn(config.batch_size, config.slow_dim).to(device),
        'au_intensities': torch.rand(config.batch_size, 16, 28).to(device),
        'target': torch.rand(config.batch_size, 3, config.image_size, config.image_size).to(device),
    }

    # Training step
    print("\nTraining step...")
    model.train()

    generated = model(
        fast_feat=batch['fast_feat'],
        slow_feat=batch['slow_feat'],
        au_intensities=batch['au_intensities'],
        apply_visual_perception=True
    )

    print(f"Generated: {generated.shape}, range=[{generated.min():.3f}, {generated.max():.3f}]")

    # Loss
    loss_l2 = loss_fns['l2'](generated, batch['target'])
    loss_perceptual = loss_fns['perceptual'](generated, batch['target'])
    loss_smooth = loss_fns['smooth'](batch['au_intensities'])
    loss_sparse = loss_fns['sparse'](model)

    loss = config.lambda_l2 * loss_l2 + \
          config.lambda_perceptual * loss_perceptual + \
          config.lambda_smooth * loss_smooth + \
          config.lambda_sparse * loss_sparse

    print(f"L2 loss: {loss_l2.item():.4f}")
    print(f"Perceptual loss: {loss_perceptual.item():.4f}")
    print(f"Smooth loss: {loss_smooth.item():.4f}")
    print(f"Sparse loss: {loss_sparse.item():.4f}")
    print(f"Total loss: {loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation step
    print("\nValidation step...")
    model.eval()
    with torch.no_grad():
        val_generated = model(
            fast_feat=batch['fast_feat'],
            slow_feat=batch['slow_feat'],
            au_intensities=batch['au_intensities'],
            apply_visual_perception=True
        )
        psnr = compute_psnr(val_generated, batch['target'])
        ssim = compute_ssim(val_generated, batch['target'])

    print(f"PSNR: {psnr:.2f}")
    print(f"SSIM: {ssim:.3f}")

    print("\n" + "=" * 60)
    print(" Quick Training Test Passed!")
    print("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        quick_test()
    else:
        train()