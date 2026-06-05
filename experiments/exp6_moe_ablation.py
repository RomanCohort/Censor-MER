"""
Experiment 6: MoE Expert Count Ablation
========================================
Test different numbers of experts to justify E=3 choice.

Tests: E=2, E=3, E=4, E=5 experts
Compare: accuracy, routing balance, specialization

Usage:
    python experiments/exp6_moe_ablation.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 6: MoE Expert Count Ablation")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# MoE with Variable Expert Count
# =============================================================================

class VariableMoE(nn.Module):
    """MoE head with configurable number of experts."""

    def __init__(self, input_dim=1280, hidden_dim=512, num_classes=4, num_experts=3):
        super().__init__()
        self.num_experts = num_experts

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        # Track last gates for analysis
        self.last_gates = None

    def forward(self, x):
        # Compute gates
        gates = self.gate(x)  # (B, E)
        self.last_gates = gates.detach()

        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        # Stack and weight
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, E, C)
        weighted = gates.unsqueeze(-1) * expert_outputs  # (B, E, C)
        output = weighted.sum(dim=1)  # (B, C)

        return output

    def get_routing_stats(self):
        """Get routing statistics per expert."""
        if self.last_gates is None:
            return None

        gates = self.last_gates  # (B, E)
        mean_per_expert = gates.mean(dim=0).cpu().numpy()
        std_per_expert = gates.std(dim=0).cpu().numpy()

        return {
            'mean': mean_per_expert,
            'std': std_per_expert,
            'balance': 1 - np.std(mean_per_expert) / (np.mean(mean_per_expert) + 1e-8),
        }


class TestModel(nn.Module):
    """Simple model for MoE testing."""

    def __init__(self, num_experts=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.moe_head = VariableMoE(input_dim=128, num_experts=num_experts)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.moe_head(feat)


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_and_evaluate(model, train_loader, test_loader, epochs=30, device='cuda'):
    """Train model and evaluate."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            video, label = batch[0], batch[1]
            video, label = video.to(device), label.to(device)

            optimizer.zero_grad()
            out = model(video)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    routing_per_class = {i: [] for i in range(4)}

    with torch.no_grad():
        for batch in test_loader:
            video, label = batch[0], batch[1]
            video, label = video.to(device), label.to(device)

            out = model(video)
            pred = out.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            # Capture routing
            gates = model.moe_head.last_gates
            for i, lbl in enumerate(label):
                routing_per_class[lbl.item()].append(gates[i].cpu().numpy())

    accuracy = correct / total * 100 if total > 0 else 0

    # Routing statistics
    routing_stats = model.moe_head.get_routing_stats()

    # Compute specialization (KL divergence from uniform)
    mean_routing_per_class = {}
    for cls, routings in routing_per_class.items():
        if routings:
            mean_routing_per_class[cls] = np.mean(routings, axis=0)

    # Overall KL
    if routing_stats:
        uniform = np.ones(model.moe_head.num_experts) / model.moe_head.num_experts
        actual = routing_stats['mean']
        kl = np.sum(actual * np.log(actual / uniform + 1e-10))
    else:
        kl = 0

    return {
        'accuracy': accuracy,
        'routing_balance': routing_stats['balance'] if routing_stats else 0,
        'kl_divergence': kl,
        'mean_routing_per_class': {k: v.tolist() for k, v in mean_routing_per_class.items()},
    }


# =============================================================================
# Expert Count Experiment
# =============================================================================

def run_expert_ablation():
    """Test different expert counts."""

    expert_counts = [2, 3, 4, 5]
    results = {}

    # Create dummy data (use real dataset in production)
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n=200):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return torch.randn(3, 16, 224, 224), torch.randint(0, 4, (1,)).item()

    train_loader = DataLoader(DummyDataset(200), batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(DummyDataset(50), batch_size=8, shuffle=False, num_workers=0)

    print("\nTesting expert counts:", expert_counts)

    for E in expert_counts:
        print(f"\n--- E = {E} ---")

        model = TestModel(num_experts=E).to(device)
        params = sum(p.numel() for p in model.parameters()) / 1e3
        print(f"  Parameters: {params:.1f}K")

        stats = train_and_evaluate(model, train_loader, test_loader, epochs=10, device=device)

        results[E] = {
            'num_experts': E,
            'params_k': params,
            **stats
        }

        print(f"  Accuracy: {stats['accuracy']:.2f}%")
        print(f"  Routing balance: {stats['routing_balance']:.3f}")
        print(f"  KL divergence: {stats['kl_divergence']:.4f}")

        # Expert utilization
        if 'mean' in stats.get('routing_stats', {}):
            print(f"  Expert utilization: {stats['routing_stats']['mean']}")

        # Cleanup immediately after each model
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        import gc
        gc.collect()  # Force garbage collection

    # Summary
    print("\n" + "=" * 60)
    print("Expert Count Comparison Summary")
    print("=" * 60)
    print(f"{'E':<5} {'Acc':<10} {'Balance':<10} {'KL':<10} {'Params':<10}")
    print("-" * 45)
    for E, r in results.items():
        print(f"{E:<5} {r['accuracy']:<10.2f} {r['routing_balance']:<10.3f} {r['kl_divergence']:<10.4f} {r['params_k']:<10.1f}K")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    results = run_expert_ablation()

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'exp6_moe_ablation.json'
    with open(output_file, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'device': str(device),
            'results': results,
        }, f, indent=2)

    print(f"\nSaved to: {output_file}")

    # Recommendation
    print("\n" + "=" * 60)
    print("Recommendation for Paper")
    print("=" * 60)

    best_E = max(results.keys(), key=lambda E: results[E]['accuracy'])
    best_balance = max(results.keys(), key=lambda E: results[E]['routing_balance'])

    print(f"Best accuracy: E={best_E} ({results[best_E]['accuracy']:.2f}%)")
    print(f"Best balance: E={best_balance} ({results[best_balance]['routing_balance']:.3f})")

    if best_E == 3:
        print("\nPaper justification: E=3 achieves best accuracy")
    else:
        print(f"\nNote: Consider updating paper to use E={best_E}")


if __name__ == '__main__':
    main()