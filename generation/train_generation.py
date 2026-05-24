# =============================================================================
# Micro-Expression Generation Training Script
# =============================================================================
# Purpose: Train the FOMM-based micro-expression generator.
#
# Training strategy:
#   1. Load pretrained FOMM
#   2. Freeze motion extractor
#   3. Fine-tune generator with AU conditions
#   4. Train AU predictor on CASME2
#
# Usage:
#   python train_generation.py --dataset casme2 --epochs 50 --batch_size 8
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_fomm import CensorFOMM
from generation.generation_loss import MicroExpressionGenerationLoss, GANLoss, Discriminator
from generation.fomm_adapter import load_pretrained_fomm


def parse_args():
    parser = argparse.ArgumentParser(description='Train Micro-Expression Generator')

    # Dataset
    parser.add_argument('--dataset', type=str, default='casme2',
                        help='Dataset to use (casme2, smic, samm)')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/data/CASME2',
                        help='Path to dataset')

    # Model
    parser.add_argument('--fomm_checkpoint', type=str, default=None,
                        help='Path to pretrained FOMM checkpoint')
    parser.add_argument('--pretrained_censor', type=str, default=None,
                        help='Path to pretrained Censor model')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')

    # GAN
    parser.add_argument('--use_gan', action='store_true',
                        help='Use GAN for adversarial training')
    parser.add_argument('--gan_mode', type=str, default='standard',
                        help='GAN mode (standard, lsgan, wgangp)')

    # Saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints/generation',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/generation',
                        help='Directory to save logs')

    return parser.parse_args()


def create_datasets(args):
    """
    Create datasets for generation training.

    For micro-expression generation, we need:
      - Neutral face (first frame)
      - AU annotations
      - Full video as target
    """
    # Placeholder - actual implementation would load CASME2
    print(f"[Dataset] Loading {args.dataset} from {args.data_root}")

    # For now, return dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Random video
            video = torch.randn(3, 16, 224, 224)
            # First frame as neutral face
            neutral_face = video[:, 0, :, :]
            # Random AU activation
            au = torch.rand(17)
            # Random emotion
            emotion = torch.randint(0, 4, (1,)).item()
            return {
                'video': video,
                'neutral_face': neutral_face,
                'au': au,
                'emotion': emotion,
            }

    dataset = DummyDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader


def train_au_predictor(model, dataloader, args, device):
    """
    Train AU predictor on CASME2.

    This is a separate phase before generation training.
    """
    print("\n[Phase 1] Training AU Predictor...")

    model.au_predictor.train()

    optimizer = optim.Adam(model.au_predictor.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    for epoch in range(args.epochs // 2):  # Half epochs for AU training
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'AU Epoch {epoch+1}'):
            video = batch['video'].to(device)
            au_target = batch['au'].to(device)
            emotion = batch['emotion'].to(device)

            # Get features from video
            features = model.feature_extractor(video) if hasattr(model, 'feature_extractor') else torch.zeros(video.shape[0], 1024).to(device)

            # Predict AU
            au_pred = model.au_predictor(features, emotion)

            # Loss
            loss = criterion(au_pred, au_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}: AU Loss = {avg_loss:.4f}")

    print("[Phase 1] AU Predictor training complete!")


def train_generator(model, discriminator, dataloader, args, device):
    """
    Train the generator with GAN.
    """
    print("\n[Phase 2] Training Generator...")

    # Optimizers
    g_optimizer = optim.Adam(model.fomm_adapter.parameters(), lr=args.lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5)

    # Loss functions
    gen_loss_fn = MicroExpressionGenerationLoss()
    gan_loss_fn = GANLoss(gan_mode=args.gan_mode)

    for epoch in range(args.epochs):
        g_loss_total = 0
        d_loss_total = 0

        for batch in tqdm(dataloader, desc=f'Gen Epoch {epoch+1}'):
            video = batch['video'].to(device)
            neutral_face = batch['neutral_face'].to(device)
            au = batch['au'].to(device)
            emotion = batch['emotion'].to(device)

            B = video.shape[0]
            num_frames = video.shape[2]

            # =================================
            # Train Generator
            # =================================
            model.fomm_adapter.train()
            discriminator.eval()

            # Generate video
            generated = model.generate(neutral_face, emotion, au, num_frames=num_frames)

            # Generation losses
            gen_losses, gen_loss = gen_loss_fn(generated, video, au, au)

            # GAN loss (fool discriminator)
            if args.use_gan:
                d_fake = discriminator(generated)
                gan_loss = gan_loss_fn(d_fake, True)  # Want discriminator to say "real"
                gen_loss += gan_loss

            g_optimizer.zero_grad()
            gen_loss.backward()
            g_optimizer.step()

            g_loss_total += gen_loss.item()

            # =================================
            # Train Discriminator (if using GAN)
            # =================================
            if args.use_gan:
                model.fomm_adapter.eval()
                discriminator.train()

                # Generate fake video
                with torch.no_grad():
                    generated = model.generate(neutral_face, emotion, au, num_frames=num_frames)

                # Real video loss
                d_real = discriminator(video)
                d_loss_real = gan_loss_fn(d_real, True)

                # Fake video loss
                d_fake = discriminator(generated)
                d_loss_fake = gan_loss_fn(d_fake, False)

                d_loss = (d_loss_real + d_loss_fake) / 2

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                d_loss_total += d_loss.item()

        # Epoch summary
        g_avg = g_loss_total / len(dataloader)
        d_avg = d_loss_total / len(dataloader) if args.use_gan else 0

        print(f"  Epoch {epoch+1}: G Loss = {g_avg:.4f}, D Loss = {d_avg:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'generator_epoch_{epoch+1}.pth')
            torch.save({
                'fomm_adapter': model.fomm_adapter.state_dict(),
                'au_predictor': model.au_predictor.state_dict() if model.au_predictor else None,
                'epoch': epoch + 1,
                'g_loss': g_avg,
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")

    print("[Phase 2] Generator training complete!")


def main():
    args = parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Setup] Using device: {device}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create model
    print("[Model] Creating Censor-FOMM...")
    model = CensorFOMM(
        fomm_checkpoint=args.fomm_checkpoint,
        device=device
    )
    model = model.to(device)

    # Create discriminator (for GAN)
    if args.use_gan:
        discriminator = Discriminator().to(device)
    else:
        discriminator = None

    # Create dataset
    dataloader = create_datasets(args)

    # Training phases
    # Phase 1: Train AU Predictor
    if model.au_predictor is not None:
        train_au_predictor(model, dataloader, args, device)

    # Phase 2: Train Generator
    train_generator(model, discriminator, dataloader, args, device)

    # Save final model
    final_path = os.path.join(args.save_dir, 'generation_final.pth')
    torch.save({
        'fomm_adapter': model.fomm_adapter.state_dict(),
        'au_predictor': model.au_predictor.state_dict() if model.au_predictor else None,
        'au_controller': model.au_controller.state_dict(),
    }, final_path)
    print(f"\n[Complete] Final model saved: {final_path}")


if __name__ == '__main__':
    main()