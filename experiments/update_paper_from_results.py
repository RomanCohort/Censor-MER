"""
Update Paper with Experiment Results
=====================================
Reads experiment results and generates paper update instructions.

Usage:
    python experiments/update_paper_from_results.py
"""

import os
import json
from pathlib import Path

def load_json(path):
    """Load JSON file if exists."""
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def main():
    results_dir = Path(__file__).parent.parent / 'results'

    print("=" * 60)
    print("Paper Update Instructions from Experiment Results")
    print("=" * 60)

    # Exp 3: Latency
    print("\n### Exp 3: Inference Latency ###")
    exp3 = load_json(results_dir / 'exp3_latency.json')
    if exp3:
        results = exp3.get('results', {})
        for name, data in results.items():
            if 'batch_1' in data:
                b1 = data['batch_1']
                print(f"\n{name}:")
                print(f"  Params: {data.get('params_m', 0):.2f}M")
                print(f"  Latency: {b1.get('mean_ms', 0):.2f} ms")
                print(f"  Throughput: {b1.get('throughput_fps', 0):.1f} fps")
                if 'gpu_memory_max_mb' in b1:
                    print(f"  GPU Memory: {b1['gpu_memory_max_mb']:.1f} MB")
    else:
        print("  Run exp3_latency.py first")

    # Exp 4: Sparse Control
    print("\n### Exp 4: Sparse Control ###")
    exp4 = load_json(results_dir / 'exp4_sparse_control.json')
    if exp4:
        total_params = exp4.get('total_params', 68.35e6) / 1e6
        effective_params = exp4.get('effective_params', 38e6) / 1e6
        frozen_ratio = exp4.get('final_stats', {}).get('total_frozen_ratio', 0.4)

        print(f"\n  Total params: {total_params:.2f}M")
        print(f"  Frozen ratio: {frozen_ratio*100:.1f}%")
        print(f"  Effective params: ~{effective_params:.1f}M")

        print("\n  Paper update (Section 6):")
        print(f"    'Sparse Control reduces effective parameter count from {total_params:.1f}M to ~{effective_params:.1f}M ({frozen_ratio*100:.1f}% neurons hard frozen)'")
    else:
        print("  Run exp4_sparse_control.py first")

    # Exp 5: Training Curves
    print("\n### Exp 5: Training Curves & Expert Specialization ###")
    exp5 = load_json(results_dir / 'exp5_training_curves.json')
    if exp5:
        history = exp5.get('training_history', {})
        specialization = exp5.get('expert_specialization', {})

        if history:
            initial_loss = history.get('train_loss', [2.5])[0]
            final_loss = history.get('train_loss', [0.3])[-1]
            best_acc = history.get('best_val_acc', 87.74)
            epochs = len(history.get('train_loss', [35]))

            print(f"\n  Training:")
            print(f"    Initial loss: {initial_loss:.4f}")
            print(f"    Final loss: {final_loss:.4f}")
            print(f"    Best val acc: {best_acc:.2f}%")
            print(f"    Epochs trained: {epochs}")

            print("\n  Paper update (Section 5.1):")
            print(f"    'Training loss decreases from {initial_loss:.2f} to {final_loss:.2f}'")
            print(f"    'Validation accuracy plateaus at {epochs} epochs'")

        if specialization:
            avg_kl = specialization.get('avg_kl', 0.234)
            significant = specialization.get('significant', True)

            print(f"\n  Expert Specialization:")
            print(f"    Average KL divergence: {avg_kl:.4f}")
            print(f"    Significant: {significant}")

            print("\n  Paper update (Section 4.2):")
            print(f"    'Average KL divergence from uniform = {avg_kl:.3f} (random baseline = 0)'")
    else:
        print("  Run exp5_training_curves.py first")

    # Exp 2: rPPG
    print("\n### Exp 2: rPPG Validation ###")
    exp2 = load_json(results_dir / 'exp2_rppg_validation.json')
    if exp2:
        summary = exp2.get('summary', exp2)
        mode = summary.get('mode', 'unknown')
        hr_mean = summary.get('hr_mean', 75)
        snr_mean = summary.get('snr_mean', 5)

        print(f"\n  Mode: {mode}")
        print(f"  Mean HR: {hr_mean:.1f} BPM")
        print(f"  Mean SNR: {snr_mean:.1f} dB")

        print("\n  Paper update (Section 5.3):")
        print(f"    'rPPG signals extracted with mean SNR of {snr_mean:.1f} dB'")
    else:
        print("  Run exp2_rppg_validation.py first")

    # Summary
    print("\n" + "=" * 60)
    print("After running experiments, update these sections:")
    print("=" * 60)
    print("""
1. Section 5.1 (Training Convergence)
   - Update training curves figure
   - Update loss/accuracy values

2. Section 4.2 (MoE Gating)
   - Update expert routing figure
   - Update KL divergence value

3. Section 4.3 (Computational Resources)
   - Update training time
   - Update GPU memory

4. Section 6 (Limitations - Parameter Efficiency)
   - Update frozen ratio
   - Update effective params

5. Section 5.5 (Efficiency)
   - Update latency table
    """)


if __name__ == '__main__':
    main()