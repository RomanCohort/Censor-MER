"""
Diagnose CASME II dataset: check Excel emotion distribution and cropped directory.
"""
import os
import sys
import pandas as pd
from pathlib import Path

DATA_ROOT = '/root/autodl-tmp/data/CASME2'

print("=" * 60)
print("CASME II Dataset Diagnosis")
print("=" * 60)

# 1. Check Excel
excel_path = os.path.join(DATA_ROOT, 'CASME2-coding-20190701.xlsx')
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = ['Subject', 'Filename', 'Unnamed2', 'OnsetFrame', 'ApexFrame',
                  'OffsetFrame', 'Unnamed6', 'ActionUnits', 'Emotion']

    print(f"\nExcel total rows: {len(df)}")
    print(f"\nEmotion distribution:")
    emotion_counts = df['Emotion'].value_counts()
    for emo, cnt in emotion_counts.items():
        print(f"  {emo}: {cnt}")

    valid_emotions = ['happiness', 'surprise', 'disgust', 'repression']
    valid_count = sum([emotion_counts.get(e, 0) for e in valid_emotions])
    print(f"\n4-class valid samples: {valid_count}")

    # 2. Check cropped directory
    cropped_dir = os.path.join(DATA_ROOT, 'cropped')
    if os.path.exists(cropped_dir):
        subjects = sorted([d for d in os.listdir(cropped_dir)
                          if os.path.isdir(os.path.join(cropped_dir, d))])
        print(f"\nCropped subjects: {len(subjects)} ({subjects[0]}-{subjects[-1]})")

        # Count actual video folders that exist
        valid_emotions_set = set(valid_emotions)
        existing_count = 0
        missing_count = 0
        per_emotion_existing = {}

        for idx, row in df.iterrows():
            subject = f"sub{int(row['Subject']):02d}"
            filename = row['Filename']
            emotion = str(row['Emotion']).strip().lower() if pd.notna(row['Emotion']) else 'others'

            if emotion not in valid_emotions_set:
                continue

            frame_dir = os.path.join(cropped_dir, subject, filename)
            if os.path.exists(frame_dir):
                n_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
                existing_count += 1
                per_emotion_existing[emotion] = per_emotion_existing.get(emotion, 0) + 1
            else:
                missing_count += 1
                if missing_count <= 10:
                    print(f"  MISSING: {frame_dir}")

        print(f"\nExisting video folders (4-class): {existing_count}")
        print(f"Missing video folders (4-class): {missing_count}")
        print(f"\nPer-emotion existing:")
        for emo in valid_emotions:
            print(f"  {emo}: {per_emotion_existing.get(emo, 0)}")
    else:
        print(f"\n[ERROR] cropped directory not found: {cropped_dir}")
else:
    print(f"[ERROR] Excel not found: {excel_path}")

# 3. Check labels.csv if exists
labels_path = os.path.join(DATA_ROOT, 'labels.csv')
if os.path.exists(labels_path):
    labels_df = pd.read_csv(labels_path)
    print(f"\nExisting labels.csv: {len(labels_df)} samples")
    print(f"  subjects: {labels_df['subject'].nunique()}")
    print(f"  labels: {labels_df['me_label'].value_counts().to_dict()}")
else:
    print(f"\nNo labels.csv found")

# 4. Check preextracted.npz
npz_path = os.path.join(DATA_ROOT, 'preextracted.npz')
if os.path.exists(npz_path):
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    print(f"\nExisting preextracted.npz: {len(data['labels'])} samples")
else:
    print(f"\nNo preextracted.npz found")

print("\n" + "=" * 60)
