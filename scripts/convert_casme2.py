"""
CASME II Frame Sequence Dataset Adapter

Converts CASME II preprocessed frame sequences to standard format.
"""

import os
import pandas as pd
from pathlib import Path

def convert_casme2_to_standard(data_root, output_csv='labels.csv'):
    """
    Convert CASME2 Excel labels to standard labels.csv format.

    Args:
        data_root: Path to CASME2 data directory
        output_csv: Output CSV filename

    Returns:
        labels.csv file with standard format
    """
    excel_path = os.path.join(data_root, 'CASME2-coding-20190701.xlsx')
    cropped_dir = os.path.join(data_root, 'cropped')

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Read Excel
    df = pd.read_excel(excel_path)

    # Clean column names
    df.columns = ['Subject', 'Filename', 'Unnamed2', 'OnsetFrame', 'ApexFrame',
                  'OffsetFrame', 'Unnamed6', 'ActionUnits', 'Emotion']

    # Emotion mapping
    emotion_map = {
        'happiness': 0,
        'sadness': 1,
        'surprise': 2,
        'fear': 3,
        'anger': 4,
        'disgust': 5,
        'contempt': 6,
        'others': 7,
        'repression': 7,
    }

    # Build samples list
    samples = []
    missing = []

    for idx, row in df.iterrows():
        subject = f"sub{int(row['Subject']):02d}"
        filename = row['Filename']

        # Construct frame directory path
        frame_dir = os.path.join(cropped_dir, subject, filename)

        if not os.path.exists(frame_dir):
            missing.append(frame_dir)
            continue

        # Get emotion label
        emotion = row['Emotion'].strip().lower() if pd.notna(row['Emotion']) else 'others'
        me_label = emotion_map.get(emotion, 7)

        # Get onset/apex/offset frames
        onset = int(row['OnsetFrame']) if pd.notna(row['OnsetFrame']) else 0
        apex = int(row['ApexFrame']) if pd.notna(row['ApexFrame']) else 0
        offset = int(row['OffsetFrame']) if pd.notna(row['OffsetFrame']) else 0

        # Count actual frames
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        num_frames = len(frame_files)

        samples.append({
            'video_path': f"{subject}/{filename}",  # Relative path
            'subject': subject,
            'filename': filename,
            'me_label': me_label,
            'emotion': emotion,
            'onset': onset,
            'apex': apex,
            'offset': offset,
            'num_frames': num_frames,
            'action_units': row['ActionUnits'] if pd.notna(row['ActionUnits']) else '',
        })

    # Save to CSV
    output_path = os.path.join(data_root, output_csv)
    samples_df = pd.DataFrame(samples)
    samples_df.to_csv(output_path, index=False)

    print(f"[CASME2 Adapter] Converted {len(samples)} samples")
    print(f"[CASME2 Adapter] Missing directories: {len(missing)}")
    if missing:
        print(f"[CASME2 Adapter] First 5 missing: {missing[:5]}")
    print(f"[CASME2 Adapter] Saved to: {output_path}")

    # Print emotion distribution
    print("\n[Emotion Distribution]")
    emotion_counts = samples_df['emotion'].value_counts()
    for emo, count in emotion_counts.items():
        print(f"  {emo}: {count}")

    return samples_df


def verify_frame_sequences(data_root):
    """
    Verify frame sequence structure and print statistics.
    """
    cropped_dir = os.path.join(data_root, 'cropped')

    if not os.path.exists(cropped_dir):
        print(f"[Error] cropped directory not found: {cropped_dir}")
        return

    subjects = sorted(os.listdir(cropped_dir))
    total_samples = 0
    total_frames = 0
    frame_counts = []

    print(f"\n[Frame Sequence Verification]")
    print(f"  Subjects: {len(subjects)}")

    for sub in subjects:
        sub_dir = os.path.join(cropped_dir, sub)
        samples = sorted(os.listdir(sub_dir))
        total_samples += len(samples)

        for sample in samples:
            sample_dir = os.path.join(sub_dir, sample)
            frames = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
            num_frames = len(frames)
            total_frames += num_frames
            frame_counts.append(num_frames)

    print(f"  Total samples: {total_samples}")
    print(f"  Total frames: {total_frames}")
    print(f"  Avg frames/sample: {total_frames / total_samples:.1f}")
    print(f"  Min frames: {min(frame_counts)}")
    print(f"  Max frames: {max(frame_counts)}")

    return {
        'subjects': len(subjects),
        'samples': total_samples,
        'frames': total_frames,
        'avg_frames': total_frames / total_samples,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/data/CASME2',
                        help='CASME2 data directory')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify, do not convert')

    args = parser.parse_args()

    if args.verify:
        verify_frame_sequences(args.data_root)
    else:
        convert_casme2_to_standard(args.data_root)
        verify_frame_sequences(args.data_root)