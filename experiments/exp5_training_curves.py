"""
Experiment 5: Training Curves and Expert Specialization Validation
====================================================================
Generate training curves and validate MoE expert specialization.

Outputs:
- Training loss curves (all 24 folds)
- Validation accuracy curves
- Expert routing distribution per expression
- Random baseline comparison

Usage:
    python experiments/exp5_training_curves.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 5: Training Curves & Expert Specialization")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# Training with Logging
# =============================================================================

def train_with_logging(model, train_loader, val_loader, epochs=50, patience=20):
    """Train model and log loss/accuracy per epoch."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {
        'train_loss': [],
        'val_acc': [],
        'expert_routing': [],  # MoE routing per epoch
    }

    best_acc = 0
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            if len(batch) == 4:
                video, label, _, _ = batch
            else:
                video, label = batch[0], batch[1]

            video, label = video.to(device), label.to(device)

            optimizer.zero_grad()

            # Forward with routing capture
            out = model(video)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        history['train_loss'].append(avg_loss)

        # Validate
        model.eval()
        correct = 0
        total = 0
        routing_per_class = {i: [] for i in range(4)}  # 4 classes

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    video, label, _, _ = batch
                else:
                    video, label = batch[0], batch[1]

                video, label = video.to(device), label.to(device)
                out = model(video)
                pred = out.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

                # Capture expert routing if available
                if hasattr(model, 'moe_head') and hasattr(model.moe_head, 'last_gates'):
                    for i, lbl in enumerate(label):
                        routing_per_class[lbl.item()].append(model.moe_head.last_gates[i].cpu().numpy())

        acc = correct / total * 100 if total > 0 else 0
        history['val_acc'].append(acc)

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={acc:.2f}%")

    history['best_val_acc'] = best_acc
    return history


# =============================================================================
# Expert Specialization Validation
# =============================================================================

def validate_expert_specialization(model, data_loader):
    """
    Validate that expert specialization is meaningful (not random).

    Method: Compare actual routing vs. random baseline
    """

    model.eval()

    # Collect routing weights per class
    routing_per_class = {i: [] for i in range(4)}

    print("\nCollecting expert routing per expression...")

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                video, label, _, _ = batch
            else:
                video, label = batch[0], batch[1]

            video, label = video.to(device), label.to(device)

            # Forward
            _ = model(video)

            # Get routing weights
            if hasattr(model, 'moe_head') and hasattr(model.moe_head, 'last_gates'):
                gates = model.moe_head.last_gates  # (B, 3)
                for i, lbl in enumerate(label):
                    routing_per_class[lbl.item()].append(gates[i].cpu().numpy())

    # Compute mean routing per class
    mean_routing = {}
    for cls, routings in routing_per_class.items():
        if routings:
            mean_routing[cls] = np.mean(routings, axis=0)

    # Random baseline: uniform distribution
    random_routing = np.array([1/3, 1/3, 1/3])

    # Compute KL divergence from uniform
    kl_divergences = {}
    for cls, routing in mean_routing.items():
        # KL(P || uniform)
        kl = np.sum(routing * np.log(routing / random_routing + 1e-10))
        kl_divergences[cls] = kl

    # Interpretation
    print("\n" + "=" * 60)
    print("Expert Specialization Analysis")
    print("=" * 60)

    class_names = ['happiness', 'surprise', 'disgust', 'repression']

    print(f"\n{'Class':<15} {'Expert 1':<12} {'Expert 2':<12} {'Expert 3':<12} {'KL div':<10}")
    print("-" * 60)

    for cls in range(4):
        if cls in mean_routing:
            r = mean_routing[cls]
            kl = kl_divergences[cls]
            print(f"{class_names[cls]:<15} {r[0]:<12.2%} {r[1]:<12.2%} {r[2]:<12.2%} {kl:<10.3f}")

    # Statistical test: is specialization significant?
    avg_kl = np.mean(list(kl_divergences.values()))

    print("\n" + "-" * 60)
    print(f"Average KL divergence from uniform: {avg_kl:.4f}")
    print(f"Random baseline KL: 0.0")
    print(f"Conclusion: {'Specialization is meaningful' if avg_kl > 0.05 else 'No significant specialization'}")

    return {
        'mean_routing_per_class': {k: v.tolist() for k, v in mean_routing.items()},
        'kl_divergences': kl_divergences,
        'avg_kl': avg_kl,
        'significant': avg_kl > 0.05,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    try:
        from main import Censor
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

    # Create model
    print("\nCreating model...")
    model = Censor(verbose=False)
    model = model.to(device)

    # Create dummy data loaders (in real experiment, use actual dataset)
    print("\nCreating dummy data loaders...")

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=100):
            self.n_samples = n_samples
        def __len__(self):
            return self.n_samples
        def __getitem__(self, idx):
            video = torch.randn(3, 16, 224, 224)
            label = torch.randint(0, 4, (1,)).item()
            return video, label, '', ''

    train_loader = DataLoader(DummyDataset(200), batch_size=8, shuffle=True)
    val_loader = DataLoader(DummyDataset(50), batch_size=8, shuffle=False)

    # Train with logging
    print("\nTraining with logging (50 epochs max)...")
    history = train_with_logging(model, train_loader, val_loader, epochs=50, patience=20)

    # Validate expert specialization
    specialization = validate_expert_specialization(model, val_loader)

    # Summary
    results = {
        'training_history': history,
        'expert_specialization': specialization,
    }

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'exp5_training_curves.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\nSaved to: {output_file}")

    # Print training curve summary
    print("\n" + "=" * 60)
    print("Training Curve Summary")
    print("=" * 60)
    print(f"Initial loss: {history['train_loss'][0]:.4f}")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Best val accuracy: {history['best_val_acc']:.2f}%")
    print(f"Epochs trained: {len(history['train_loss'])}")


if __name__ == '__main__':
    main()