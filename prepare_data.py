# =============================================================================
# Censor -- Data Preparation Guide
# =============================================================================
# Instructions for downloading and preparing micro-expression datasets.
#
# All datasets require signed license agreements. Download links and
# preparation instructions are provided below.
#
# Usage:
#     python prepare_data.py --dataset casme2 --output_dir ./data
#
# After preparation, the folder structure should be:
#     ./data/<dataset>/
#         videos/
#             video1.avi
#             video2.avi
#         labels.csv
# =============================================================================

import os
import sys
import argparse
import csv
import subprocess
import urllib.request

# =============================================================================
# Dataset Information
# =============================================================================

DATASETS = {
    'casme2': {
        'name': 'CASME II',
        'full_name': 'CASME II Micro-Expression Database',
        'url': 'http://casme.psych.ac.cn/casme/c2',
        'papers': [
            'Yan et al., "CASME II: An Improved Spontaneous Micro-Expression Database"',
            'IEEE Transactions on Affective Computing, 2014.',
            'PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3903513/',
        ],
        'samples': 247,
        'subjects': 26,
        'fps': 200,
        'resolution': '640×480',
        'emotion_classes': 5,
        'format': 'AVI (MJPG)',
        'license_required': True,
        'expected_structure': {
            'video_dir': 'cropped',
            'annotation_file': 'CASME2.xlsx or label.txt',
        },
        'notes': [
            'Download from http://fu.psych.ac.cn/CASME/casme2-en.php',
            'Sign the license agreement at https://www.wjx.top/vj/hSaLoan.aspx',
            'Videos are pre-cropped to face regions (~280×340 pixels)',
            'Labels include: subject ID, video filename, onset/peak/offset frames, emotion label',
            'Convert emotion labels to numeric: 1=happy, 2=sadness, 3=surprise, 4=fear, 5=anger, 6=disgust, 7=contempt, 0=others',
        ],
    },
    'samm': {
        'name': 'SAMM',
        'full_name': 'Spontaneous Actions and Micro-Movement Database',
        'url': 'https://www.mmu.ac.uk',
        'papers': [
            'Davison et al., "SAMM: A Spontaneous Micro-Facial Movement Database"',
            'IEEE Transactions on Affective Computing, 2018.',
        ],
        'samples': 159,
        'subjects': 32,
        'fps': 200,
        'resolution': '2040×1088',
        'emotion_classes': 7,
        'format': 'AVI (MJPG)',
        'license_required': True,
        'expected_structure': {
            'video_dir': 'videos',
            'annotation_file': 'SAMM_labels.csv',
        },
        'notes': [
            'Contact A.K. Davison at Manchester Metropolitan University',
            '13 different ethnicities (most diverse dataset)',
            'Labels include: subject ID, video filename, onset/peak/offset frames, AU labels, emotion',
        ],
    },
    'smic': {
        'name': 'SMIC',
        'full_name': 'Spontaneous Micro-expression Database',
        'url': 'https://www.oulu.fi',
        'papers': [
            'Pfister et al., "MIC: Spontaneous Micro-expression Database"',
            'IEEE International Conference on Automatic Face and Gesture Recognition, 2013.',
        ],
        'samples': 164,
        'subjects': 16,
        'fps': 100,  # HS subset
        'resolution': '640×480',
        'emotion_classes': 3,
        'format': 'AVI (MJPG)',
        'license_required': True,
        'expected_structure': {
            'video_dir': 'HS/videos',
            'annotation_file': 'SMIC_labels.xlsx',
        },
        'notes': [
            'Download from University of Oulu, Finland',
            'Three subsets: HS (100fps), VIS (25fps), NIR (25fps)',
            'We recommend using HS (high-speed) subset for best quality',
            'Labels: 1=positive, 2=negative, 3=surprise',
        ],
    },
    'mmew': {
        'name': 'MMEW',
        'full_name': 'Micro-Expression in the Wild',
        'url': 'https://github.com/benxianyeteam/MMEW-Dataset',
        'papers': [
            'Nie et al., "MMEW: A Micro-Expression Database with Multi-modal Labels"',
            'IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022.',
        ],
        'samples': 300,  # micro-expressions
        'subjects': 36,
        'fps': 90,
        'resolution': '1920×1080',
        'emotion_classes': 7,
        'format': 'MP4',
        'license_required': True,  # Academic use only
        'expected_structure': {
            'video_dir': 'Micro-expression',
            'annotation_file': 'labels.csv',
        },
        'notes': [
            'GitHub: https://github.com/benxianyeteam/MMEW-Dataset',
            'Contains both micro-expressions (300) and macro-expressions (900)',
            'High resolution (1920×1080), facial region ~400×400',
            'Can use macro expressions for pre-training or data augmentation',
            'Labels include: subject ID, video filename, emotion label, AU labels',
        ],
    },
    'casme3': {
        'name': 'CAS(ME)³',
        'full_name': 'CAS(ME)³ — A Multi-modal Microscopic Expression Database',
        'url': 'http://melab.psych.ac.cn',
        'papers': [
            'Qu et al., "CAS(ME)³: A Third Generation Microscopic Expression Database"',
            'IEEE Transactions on Affective Computing, 2022.',
        ],
        'samples': 300,
        'fps': 30,
        'resolution': 'Various',
        'emotion_classes': 4,
        'format': 'AVI',
        'license_required': True,
        'expected_structure': {
            'video_dir': 'videos',
            'annotation_file': 'CASME3_labels.csv',
        },
        'notes': [
            'Part of the CASME series (CASME, CASME II, CASME³)',
            'Includes both spotting (long video) and recognition tasks',
            'Labels include: subject ID, video filename, onset/peak/offset, emotion, AUs',
        ],
    },
}

# =============================================================================
# Annotation CSV Format
# =============================================================================

CSV_HEADERS = [
    'video_path',
    'subject',
    'me_label',
    'au_01', 'au_02', 'au_03', 'au_04', 'au_05', 'au_06', 'au_07', 'au_08',
    'au_09', 'au_10', 'au_11', 'au_12', 'au_13', 'au_14', 'au_15', 'au_16',
    'au_17', 'au_18', 'au_19', 'au_20', 'au_21', 'au_22', 'au_23', 'au_24',
    'au_25', 'au_26', 'au_27', 'au_28',
]

# 7-class ME label mapping (consistent across all datasets)
ME_LABELS = {
    'happiness': 0, 'happy': 0, '1': 0, 'positive': 0,
    'sadness': 1, 'sad': 1, '2': 1, 'negative': 1,
    'surprise': 2, '3': 2,
    'fear': 3, '4': 3,
    'anger': 4, '5': 4,
    'disgust': 5, '6': 5,
    'contempt': 6, '7': 6, 'repression': 6,
    'others': -1, '0': -1,
}


# =============================================================================
# Preparation Functions
# =============================================================================

def convert_label(label_str):
    """Convert emotion label string to numeric (0-6) or -1 for unknown."""
    label_str = str(label_str).strip().lower()
    return ME_LABELS.get(label_str, -1)


def prepare_labels_csv(dataset_name, data_dir, output_csv_path):
    """
    Convert dataset-specific annotation format to unified CSV format.

    This is a template function — specific implementation depends on
    the dataset's original annotation format.

    For CASME II, a typical label.txt looks like:
        video_name, subject, onset, apex, offset, emotion_code

    For SAMM, a CSV like:
        Subject, Video, Onset, Apex, Offset, Emotion, AU1, AU2, ...

    The function below provides a generic template. Adjust based on
    your dataset's actual annotation format.
    """
    print(f"[prepare_labels] Creating labels.csv for {dataset_name}...")

    # Check for annotation file
    annotation_files = [
        'labels.csv', 'annotations.csv', 'label.txt',
        'CASME2_label.txt', 'SAMM_labels.csv', 'SMIC_labels.xlsx'
    ]

    anno_path = None
    for fname in annotation_files:
        candidate = os.path.join(data_dir, fname)
        if os.path.exists(candidate):
            anno_path = candidate
            break

    if anno_path is None:
        print(f"  WARNING: No annotation file found. Creating empty labels.csv...")
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        return

    print(f"  Found annotation: {anno_path}")

    # Parse annotation (template — adjust per dataset)
    # This is a placeholder implementation. The actual parsing depends
    # on the annotation format of each dataset.
    samples = []

    # Check file extension
    if anno_path.endswith('.csv'):
        with open(anno_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Adapt column names here based on actual annotation format
                video_path = row.get('video_path') or row.get('filename') or row.get('Video', '')
                subject = row.get('subject') or row.get('Subject', '')
                me_label = convert_label(row.get('emotion') or row.get('Emotion', 0))

                au_row = {}
                for i in range(1, 29):
                    key = f'au_{i:02d}'
                    val = row.get(key, 0)
                    if val in ['1', 'true', 'True', 1]:
                        au_row[key] = 1.0
                    else:
                        try:
                            au_row[key] = float(val)
                        except (ValueError, TypeError):
                            au_row[key] = 0.0

                sample = {
                    'video_path': video_path,
                    'subject': subject,
                    'me_label': me_label,
                    **au_row,
                }
                samples.append(sample)
    else:
        # For .txt or other formats, use a generic parser
        print(f"  WARNING: Non-CSV annotation format. Manual processing required.")

    # Write unified CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for sample in samples:
            # Add missing AU columns with default 0
            for i in range(1, 29):
                key = f'au_{i:02d}'
                if key not in sample:
                    sample[key] = 0.0
            writer.writerow(sample)

    print(f"  Labels saved to: {output_csv_path}")
    print(f"  Total samples: {len(samples)}")


# =============================================================================
# Dataset Download & Preparation
# =============================================================================

def download_and_prepare(dataset_name, output_dir):
    """Guide user through dataset download and preparation."""
    info = DATASETS[dataset_name]
    dataset_dir = os.path.join(output_dir, dataset_name)

    print(f"\n{'='*60}")
    print(f" Preparing Dataset: {info['name']}")
    print(f"{'='*60}")
    print(f"\n  Full name: {info['full_name']}")
    print(f"  URL: {info['url']}")
    print(f"  Samples: {info['samples']} | Subjects: {info['subjects']} | FPS: {info['fps']}")
    print(f"  Format: {info['format']} | Resolution: {info['resolution']}")
    print(f"  Emotion classes: {info['emotion_classes']}")
    print(f"  License required: {'Yes' if info['license_required'] else 'No'}")

    print(f"\n{'='*60}")
    print(f" Download Instructions")
    print(f"{'='*60}")
    for note in info['notes']:
        print(f"  - {note}")

    print(f"\n{'='*60}")
    print(f" Expected Folder Structure")
    print(f"{'='*60}")
    print(f"  {dataset_dir}/")
    print(f"    videos/")
    print(f"      video1.avi  (or .mp4)")
    print(f"      video2.avi")
    print(f"      ...")
    print(f"    labels.csv  (generated by this script)")

    print(f"\n{'='*60}")
    print(f" After downloading:")
    print(f"{'='*60}")
    print(f"  1. Place all video files in: {os.path.join(dataset_dir, 'videos')}/")
    print(f"  2. Place annotation file in: {dataset_dir}/")
    print(f"  3. Run: python prepare_data.py --dataset {dataset_name} --generate-labels")
    print(f"  4. Verify: python dataset.py --root {dataset_dir}")
    print(f"  5. Train: python train.py --dataset {dataset_name} --data_root {output_dir}")

    # Create directories
    os.makedirs(os.path.join(dataset_dir, 'videos'), exist_ok=True)

    return dataset_dir


# =============================================================================
# Generate Synthetic Labels (for testing without real data)
# =============================================================================

def generate_synthetic_labels(dataset_name, output_dir, num_samples=50):
    """Generate synthetic labels.csv for testing the pipeline."""
    import random

    dataset_dir = os.path.join(output_dir, dataset_name)
    videos_dir = os.path.join(dataset_dir, 'videos')

    print(f"[generate_synthetic] Creating synthetic labels.csv in {dataset_dir}...")

    # Find video files
    video_files = []
    for ext in ['*.avi', '*.mp4', '*.AVI', '*.MP4']:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))

    if not video_files:
        print(f"  WARNING: No video files found in {videos_dir}")
        print(f"  Creating synthetic entries with placeholder paths...")

        # Create synthetic entries without actual videos
        rows = []
        for i in range(num_samples):
            subject_id = f"s{i // 5 + 1:03d}"  # ~5 videos per subject
            video_name = f"sample_{i:04d}.avi"
            me_label = random.randint(0, 6)
            au_vals = [random.randint(0, 1) if random.random() > 0.7 else 0 for _ in range(28)]

            row = {
                'video_path': video_name,
                'subject': subject_id,
                'me_label': me_label,
            }
            for j in range(1, 29):
                row[f'au_{j:02d}'] = au_vals[j - 1]
            rows.append(row)
    else:
        print(f"  Found {len(video_files)} video files.")
        rows = []
        for vf in video_files:
            subject_id = os.path.basename(os.path.dirname(vf))
            video_name = os.path.basename(vf)
            me_label = random.randint(0, 6)
            au_vals = [random.randint(0, 1) if random.random() > 0.7 else 0 for _ in range(28)]

            row = {
                'video_path': video_name,
                'subject': subject_id,
                'me_label': me_label,
            }
            for j in range(1, 29):
                row[f'au_{j:02d}'] = au_vals[j - 1]
            rows.append(row)

    # Write CSV
    csv_path = os.path.join(dataset_dir, 'labels.csv')
    os.makedirs(dataset_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Labels saved: {csv_path} ({len(rows)} samples)")
    return csv_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Censor Data Preparation — Download and prepare MER datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='casme2',
                        choices=list(DATASETS.keys()),
                        help='Dataset to prepare')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Root output directory')
    parser.add_argument('--generate-labels', action='store_true',
                        help='Generate labels.csv from annotation files')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic labels for pipeline testing')
    parser.add_argument('--info', action='store_true',
                        help='Show dataset information without downloading')

    args = parser.parse_args()

    if args.info:
        # Show info for all datasets
        for name, info in DATASETS.items():
            print(f"\n{name}: {info['name']}")
            print(f"  URL: {info['url']}")
            print(f"  Samples: {info['samples']}, Subjects: {info['subjects']}")
            print(f"  FPS: {info['fps']}, Resolution: {info['resolution']}")
            print(f"  License required: {info['license_required']}")
        return

    if args.synthetic:
        # Generate synthetic labels for testing
        generate_synthetic_labels(args.dataset, args.output_dir, num_samples=50)
        print("\nSynthetic labels generated.")
        print("To test the pipeline:")
        print(f"  python dataset.py --root {os.path.join(args.output_dir, args.dataset)}")
        return

    # Show preparation instructions
    download_and_prepare(args.dataset, args.output_dir)

    # Generate labels if annotation files exist
    if args.generate_labels:
        dataset_dir = os.path.join(args.output_dir, args.dataset)
        csv_path = os.path.join(dataset_dir, 'labels.csv')
        prepare_labels_csv(args.dataset, dataset_dir, csv_path)

    # Summary
    print(f"\n{'='*60}")
    print(f" Summary")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output directory: {os.path.join(args.output_dir, args.dataset)}")
    print(f"\n  Next steps:")
    print(f"  1. Download data from {DATASETS[args.dataset]['url']}")
    print(f"  2. Sign license agreement if required")
    print(f"  3. Place videos in: {os.path.join(args.output_dir, args.dataset, 'videos')}/")
    print(f"  4. Run: python prepare_data.py --dataset {args.dataset} --generate-labels")
    print(f"  5. Verify: python dataset.py --root {os.path.join(args.output_dir, args.dataset)}")
    print(f"  6. Train: python train.py --dataset {args.dataset} --data_root {args.output_dir}")


if __name__ == '__main__':
    main()