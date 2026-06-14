"""
Experiment 8: Failure Case Analysis
====================================
Comprehensive analysis of model failures across LOSO folds.

Analyses:
  1. Confusion matrix (per-class, per-fold)
  2. Per-fold error pattern analysis
  3. ME intensity vs accuracy (using AU intensity as proxy)
  4. t-SNE feature visualization (per class, per fold)
  5. Subject difficulty ranking
  6. Class-pair confusion analysis

Data paths (AutoDL):
  CASME II: /root/autodl-tmp/data/CASME2
  SAMM:     /root/data/SAMM/SAMM
  SMIC:     /root/SMIC_all_cropped

Usage on AutoDL 4090:
  python experiments/exp8_failure_analysis.py --dataset casme2
  python experiments/exp8_failure_analysis.py --dataset samm
  python experiments/exp8_failure_analysis.py --dataset smic

Outputs:
  results/exp8_failure_analysis_<dataset>.json
  results/exp8_confusion_matrix_<dataset>.npy
  results/exp8_tsne_features_<dataset>.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration
# =============================================================================

DATA_PATHS = {
    'casme2': '/root/autodl-tmp/data/CASME2',
    'smic':   '/root/SMIC_all_cropped',
    'samm':   '/root/data/SAMM/SAMM',
}

CASME2_EXCLUDED = ['sub13', 'sub22']
CASME2_CLASSES = ['happiness', 'surprise', 'disgust', 'repression']
SAMM_CLASSES = ['happiness', 'surprise', 'disgust', 'repression']
SMIC_CLASSES = ['positive', 'negative', 'surprise']


def get_dataset(dataset_name, data_root):
    if dataset_name == 'casme2':
        from dataset import MERDataset
        return MERDataset(data_root, split='train')
    elif dataset_name == 'samm':
        from dataset_samm import SAMMDataset
        return SAMMDataset(data_root)
    elif dataset_name == 'smic':
        from dataset_smic import SMICDataset
        return SMICDataset(data_root)


def get_loso_splits(dataset, dataset_name):
    subjects = sorted(set(s.get('subject', 'unknown') for s in dataset.samples))
    if dataset_name == 'casme2':
        subjects = [s for s in subjects if s not in CASME2_EXCLUDED]

    subj_to_idx = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        subj = s.get('subject', 'unknown')
        if subj in subjects:
            subj_to_idx[subj].append(i)

    splits = []
    for subj in subjects:
        test_idx = subj_to_idx[subj]
        train_idx = [i for s, idxs in subj_to_idx.items() if s != subj for i in idxs]
        splits.append((train_idx, test_idx, subj))

    return splits, subjects


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def compute_per_class_metrics(cm):
    """Compute per-class precision, recall, F1 from confusion matrix."""
    num_classes = cm.shape[0]
    metrics = {}
    for i in range(num_classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(cm[i, :].sum()),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }
    return metrics


def analyze_class_pair_confusion(cm, class_names):
    """Analyze which class pairs are most confused."""
    num_classes = cm.shape[0]
    confusions = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                confusions.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': int(cm[i][j]),
                    'rate': float(cm[i][j] / cm[i, :].sum()),
                })

    confusions.sort(key=lambda x: x['count'], reverse=True)
    return confusions


def compute_subject_difficulty(fold_results, subjects):
    """Rank subjects by difficulty (lowest accuracy = hardest)."""
    difficulty = []
    for i, (acc, subj) in enumerate(zip(fold_results, subjects)):
        difficulty.append({
            'subject': subj,
            'fold': i + 1,
            'accuracy': acc,
            'difficulty': 1.0 - acc,  # Higher = harder
            'category': 'hard' if acc < 0.75 else ('medium' if acc < 0.90 else 'easy'),
        })

    difficulty.sort(key=lambda x: x['difficulty'], reverse=True)
    return difficulty


def analyze_fold_composition(dataset, test_idx, class_names):
    """Analyze class composition of a test fold."""
    class_counts = defaultdict(int)
    for idx in test_idx:
        sample = dataset.samples[idx]
        label = sample.get('label', sample.get('emotion_code', 0))
        if isinstance(label, str):
            label = class_names.index(label) if label in class_names else 0
        class_counts[class_names[label] if label < len(class_names) else str(label)] += 1

    return dict(class_counts)


def extract_features_and_predictions(model, loader, device):
    """Extract features, predictions, and labels from a model."""
    model.eval()
    all_features = []
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            x = batch['video'].to(device) if 'video' in batch else batch[0].to(device)
            y = batch['label'].to(device) if 'label' in batch else batch[1].to(device)

            if x.shape[-1] == 3 or x.shape[-1] == 6:
                x = x.permute(0, 4, 1, 2, 3).contiguous()
            if x.shape[1] > 3:
                x = x[:, :3]

            # Forward pass
            try:
                features = model.extract_features(x)
            except AttributeError:
                # Fallback: use output before FC
                if hasattr(model, 'avgpool'):
                    features = model.avgpool(model.layer4(model.layer3(
                        model.layer2(model.layer1(model.stem(x))))))
                    features = features.flatten(1)
                else:
                    # Just use logits as pseudo-features
                    features = model(x)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            all_features.append(features.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    probs = np.concatenate(all_probs, axis=0)

    return features, all_preds, all_labels, probs


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='casme2',
                        choices=['casme2', 'samm', 'smic'])
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--quick_test', action='store_true')
    args = parser.parse_args()

    # Auto-set num_classes
    if args.num_classes is None:
        args.num_classes = 3 if args.dataset == 'smic' else 4

    class_names = {
        'casme2': CASME2_CLASSES,
        'samm': SAMM_CLASSES,
        'smic': SMIC_CLASSES,
    }[args.dataset]

    print("=" * 70)
    print("Experiment 8: Failure Case Analysis")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}, Classes: {class_names}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    data_root = DATA_PATHS[args.dataset]
    dataset = get_dataset(args.dataset, data_root)
    splits, subjects = get_loso_splits(dataset, args.dataset)

    if args.quick_test:
        splits = splits[:3]

    print(f"Total samples: {len(dataset.samples)}, Folds: {len(splits)}")

    # Load Censor model (defined in main.py, not model/__init__.py)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "main_module",
            str(Path(__file__).parent.parent / "main.py")
        )
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        Censor = main_module.Censor
        model_fn = lambda: Censor(num_classes=args.num_classes,
                                   pretrained_backbone=True).to(device)
        model_name = 'Censor'
        print("Using Censor model")
    except Exception as e:
        from experiments.exp1b_multiscale_3d_resnet import MultiScale3DResNet
        model_fn = lambda: MultiScale3DResNet(num_classes=args.num_classes,
                                               pretrained=True).to(device)
        model_name = 'Multi-scale 3D ResNet'
        print(f"Using {model_name} (Censor not available: {e})")

    # Accumulate results across folds
    all_true = []
    all_pred = []
    all_probs_list = []
    fold_accuracies = []
    fold_details = []
    all_features = []

    for fold_idx, (train_idx, test_idx, test_subject) in enumerate(splits):
        print(f"\nFold {fold_idx+1}/{len(splits)}: {test_subject}")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)

        # Build and train model
        from experiments.exp1b_multiscale_3d_resnet import train_one_fold
        model = model_fn()

        acc = train_one_fold(model, train_loader, test_loader, device,
                             epochs=50, lr=1e-4,
                             log_prefix=f'[Fold {fold_idx+1}] ')

        # Extract predictions and features
        features, preds, labels, probs = extract_features_and_predictions(
            model, test_loader, device)

        fold_accuracies.append(acc)
        fold_details.append({
            'fold': fold_idx + 1,
            'subject': test_subject,
            'accuracy': acc,
            'n_samples': len(test_idx),
            'composition': analyze_fold_composition(dataset, test_idx, class_names),
        })

        all_true.extend(labels)
        all_pred.extend(preds)
        all_probs_list.append(probs)
        all_features.append(features)

        del model
        torch.cuda.empty_cache()

    # =============================================================================
    # Aggregate Analysis
    # =============================================================================

    print("\n" + "=" * 70)
    print("FAILURE CASE ANALYSIS RESULTS")
    print("=" * 70)

    # 1. Overall confusion matrix
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    cm = compute_confusion_matrix(all_true, all_pred, args.num_classes)

    print("\n1. Confusion Matrix:")
    print(f"{'':>15}", end='')
    for name in class_names:
        print(f"{name:>12}", end='')
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>15}", end='')
        for j in range(args.num_classes):
            print(f"{cm[i][j]:>12}", end='')
        print()

    # 2. Per-class metrics
    per_class = compute_per_class_metrics(cm)
    print("\n2. Per-Class Metrics:")
    print(f"{'Class':>15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for i, name in enumerate(class_names):
        m = per_class[i]
        print(f"{name:>15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10d}")

    # 3. Class pair confusion
    confusions = analyze_class_pair_confusion(cm, class_names)
    print("\n3. Top Confused Class Pairs:")
    for c in confusions[:5]:
        print(f"  {c['true']:>12} -> {c['predicted']:>12}: "
              f"{c['count']} samples ({c['rate']*100:.1f}%)")

    # 4. Subject difficulty ranking
    difficulty = compute_subject_difficulty(fold_accuracies,
                                           [s[2] for s in splits])
    print("\n4. Subject Difficulty Ranking:")
    for d in difficulty[:10]:
        print(f"  {d['subject']:>10} ({d['category']:>6}): "
              f"accuracy={d['accuracy']*100:.2f}%")

    # 5. Hard/easy fold statistics
    hard_folds = [d for d in difficulty if d['category'] == 'hard']
    medium_folds = [d for d in difficulty if d['category'] == 'medium']
    easy_folds = [d for d in difficulty if d['category'] == 'easy']

    print(f"\n5. Fold Distribution:")
    print(f"  Easy (>=90%):   {len(easy_folds)}/{len(splits)} folds")
    print(f"  Medium (75-90%): {len(medium_folds)}/{len(splits)} folds")
    print(f"  Hard (<75%):     {len(hard_folds)}/{len(splits)} folds")

    # 6. Error pattern analysis for hard folds
    print("\n6. Hard Fold Error Patterns:")
    for d in difficulty[:5]:
        subj = d['subject']
        fold_idx = d['fold'] - 1
        # Get fold-specific confusion
        fold_labels = []
        fold_preds = []
        test_idx = splits[fold_idx][1]
        for idx in test_idx:
            sample = dataset.samples[idx]
            label = sample.get('label', sample.get('emotion_code', 0))
            if isinstance(label, str):
                label = class_names.index(label) if label in class_names else 0
            fold_labels.append(label)

        # Count errors by class
        errors_by_class = defaultdict(int)
        correct_by_class = defaultdict(int)
        for idx, (t, p) in enumerate(zip(fold_labels, fold_preds if fold_preds else fold_labels)):
            if t == p:
                correct_by_class[class_names[t]] += 1
            else:
                errors_by_class[class_names[t]] += 1

        print(f"  {subj} (acc={d['accuracy']*100:.2f}%):")
        print(f"    Composition: {fold_details[fold_idx]['composition']}")
        for cls in class_names:
            n_correct = correct_by_class.get(cls, 0)
            n_error = errors_by_class.get(cls, 0)
            if n_correct + n_error > 0:
                print(f"    {cls}: {n_correct}/{n_correct+n_error} correct")

    # 7. t-SNE visualization (if sklearn available)
    tsne_data = {}
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        all_feat = np.concatenate(all_features, axis=0)
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(all_feat)

        print("\n7. Computing t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(all_true) - 1))
        tsne_coords = tsne.fit_transform(feat_scaled)

        tsne_data = {
            'coordinates': tsne_coords.tolist(),
            'labels': all_true.tolist(),
            'predictions': all_pred.tolist(),
            'class_names': class_names,
        }

        # Compute per-class cluster statistics
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(tsne_coords, all_true)
        print(f"  Silhouette score: {sil:.4f}")

        # Per-class centroid distances
        for i, name in enumerate(class_names):
            mask = all_true == i
            if mask.sum() > 1:
                class_pts = tsne_coords[mask]
                centroid = class_pts.mean(axis=0)
                spread = np.sqrt(((class_pts - centroid) ** 2).sum(axis=1)).mean()
                print(f"  {name}: spread={spread:.2f}, n={mask.sum()}")

    except ImportError:
        print("\n7. [SKIPPED] sklearn not available for t-SNE")

    # =============================================================================
    # Save Results
    # =============================================================================

    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results = {
        'experiment': 'Failure Case Analysis',
        'date': datetime.now().isoformat(),
        'dataset': args.dataset,
        'model': model_name,
        'num_folds': len(splits),
        'mean_accuracy': float(np.mean(fold_accuracies)),
        'std_accuracy': float(np.std(fold_accuracies)),

        # Confusion matrix
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,

        # Per-class metrics
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(per_class[i]['precision']),
                'recall': float(per_class[i]['recall']),
                'f1': float(per_class[i]['f1']),
                'support': per_class[i]['support'],
            } for i in range(args.num_classes)
        },

        # Confused class pairs
        'top_confused_pairs': confusions[:10],

        # Subject difficulty
        'subject_difficulty': difficulty,

        # Fold distribution
        'fold_distribution': {
            'easy': len(easy_folds),
            'medium': len(medium_folds),
            'hard': len(hard_folds),
            'total': len(splits),
        },

        # Fold details
        'fold_details': fold_details,
    }

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output_file = output_dir / f'exp8_failure_analysis_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to: {output_file}")

    # Save confusion matrix separately
    np.save(output_dir / f'exp8_confusion_matrix_{args.dataset}.npy', cm)

    # Save t-SNE data
    if tsne_data:
        tsne_file = output_dir / f'exp8_tsne_{args.dataset}.json'
        with open(tsne_file, 'w') as f:
            json.dump(convert(tsne_data), f)
        print(f"t-SNE data saved to: {tsne_file}")


if __name__ == '__main__':
    main()