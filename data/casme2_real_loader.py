# =============================================================================
# CASME2 Real Dataset Loader for Generation Training
# =============================================================================
# 从CASME2数据集加载真实微表情样本用于生成训练
#
# 数据结构：
#   CASME2_RAW/images/subject/video/frame.jpg
#   CASME2_labeling.xlsx: AU标注 + 情感类别
#
# 输出：
#   - neutral_face: 起始帧（中性脸）
#   - target_video: 微表情视频序列
#   - au_activation: AU强度标注
#   - emotion_class: 情感类别
#   - intensity: 微表情强度
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import cv2
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_snn import AU_INDEX

# AU到CASME2标注的映射
CASME2_AU_MAPPING = {
    'AU1': 'AU1',
    'AU2': 'AU2',
    'AU4': 'AU4',
    'AU5': 'AU5',
    'AU6': 'AU6',
    'AU7': 'AU7',
    'AU9': 'AU9',
    'AU10': 'AU10',
    'AU12': 'AU12',
    'AU14': 'AU14',
    'AU15': 'AU15',
    'AU17': 'AU17',
    'AU20': 'AU20',
    'AU23': 'AU23',
    'AU24': 'AU24',
    'AU25': 'AU25',
    'AU26': 'AU26',
}

# CASME2情感类别映射
CASME2_EMOTION_MAPPING = {
    'happiness': 0,
    'surprise': 1,
    'disgust': 2,
    'repression': 3,
    # 其他类别映射到相近类别
    'fear': 1,        # 映射到surprise
    'sadness': 3,     # 映射到repression
    'others': 3,
    'neutral': -1,    # 不使用
}


class CASME2RealDataset(Dataset):
    """
    CASME2真实微表情数据集

    用于Censor-G SNN生成训练：
      - neutral_face: 起始帧作为中性脸输入
      - target_video: 整个微表情序列作为生成目标
      - au_activation: 从FACS标注提取的AU强度
      - emotion_class: 情感类别索引
    """

    def __init__(self,
                 data_root: str,
                 annotation_file: str = None,
                 image_size: int = 224,
                 num_frames: int = 16,
                 emotion_filter: List[str] = ['happiness', 'surprise', 'disgust', 'repression'],
                 min_frames: int = 8,
                 normalize: bool = True):
        """
        Args:
            data_root: CASME2数据根目录
            annotation_file: 标注文件路径（xlsx或csv）
            image_size: 输出图像尺寸
            num_frames: 视频帧数（会截断或填充）
            emotion_filter: 只使用这些情感的样本
            min_frames: 最小帧数要求
            normalize: 是否归一化到[0,1]
        """
        self.data_root = data_root
        self.image_size = image_size
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.normalize = normalize
        self.emotion_filter = emotion_filter

        # 加载标注
        if annotation_file:
            self.annotations = self._load_annotations(annotation_file)
        else:
            # 尝试默认路径
            default_path = os.path.join(data_root, 'CASME2_labeling.xlsx')
            if os.path.exists(default_path):
                self.annotations = self._load_annotations(default_path)
            else:
                # 尝试CSV
                csv_path = os.path.join(data_root, 'CASME2_labeling.csv')
                if os.path.exists(csv_path):
                    self.annotations = self._load_annotations_csv(csv_path)
                else:
                    # 使用模拟标注
                    self.annotations = self._create_mock_annotations(data_root)

        # 过滤样本
        self.samples = self._filter_samples()

        print(f"[CASME2RealDataset] Loaded {len(self.samples)} valid samples")
        print(f"  Emotions: {emotion_filter}")
        print(f"  Image size: {image_size}")
        print(f"  Num frames: {num_frames}")

    def _load_annotations(self, xlsx_path):
        """加载xlsx标注文件"""
        try:
            df = pd.read_excel(xlsx_path)

            # CASME2标注格式（可能需要调整）
            # 常见列: subject, video, emotion, onset, apex, offset, AU标注

            annotations = []
            for _, row in df.iterrows():
                sample = {
                    'subject': str(row.get('subject', '')),
                    'video': str(row.get('video', row.get('filename', ''))),
                    'emotion': str(row.get('emotion', row.get('Emotion', 'others'))),
                    'onset': int(row.get('onset', row.get('OnsetFrame', 0))),
                    'apex': int(row.get('apex', row.get('ApexFrame', 0))),
                    'offset': int(row.get('offset', row.get('OffsetFrame', 0))),
                    'au_annotations': self._parse_au_row(row),
                }

                # 添加帧数
                sample['num_frames'] = sample['offset'] - sample['onset'] + 1

                annotations.append(sample)

            return annotations

        except Exception as e:
            print(f"[Warning] Could not load xlsx: {e}")
            return None

    def _load_annotations_csv(self, csv_path):
        """加载csv标注文件"""
        try:
            df = pd.read_csv(csv_path)
            return self._parse_dataframe(df)
        except Exception as e:
            print(f"[Warning] Could not load csv: {e}")
            return None

    def _parse_dataframe(self, df):
        """解析DataFrame为标注列表"""
        annotations = []
        for _, row in df.iterrows():
            sample = {
                'subject': str(row.get('subject', '')),
                'video': str(row.get('video', row.get('filename', ''))),
                'emotion': str(row.get('emotion', 'others')),
                'onset': int(row.get('onset', 0)),
                'apex': int(row.get('apex', 0)),
                'offset': int(row.get('offset', 0)),
                'au_annotations': self._parse_au_row(row),
                'num_frames': int(row.get('offset', 0)) - int(row.get('onset', 0)) + 1,
            }
            annotations.append(sample)
        return annotations

    def _parse_au_row(self, row):
        """从行数据解析AU标注"""
        au_dict = {}

        for au_name, casme2_name in CASME2_AU_MAPPING.items():
            # 尝试多种列名格式
            intensity = None
            for col_format in [casme2_name, f'{casme2_name}_intensity', f'AU{au_name[2:]}', au_name]:
                if col_format in row:
                    intensity = row[col_format]
                    break

            if intensity is not None:
                try:
                    au_dict[AU_INDEX[au_name]] = float(intensity) / 5.0  # CASME2通常用1-5强度
                except:
                    au_dict[AU_INDEX[au_name]] = 0.5

        return au_dict

    def _create_mock_annotations(self, data_root):
        """创建模拟标注（当无真实标注时）"""
        annotations = []

        # 尝试多种可能的目录名
        possible_dirs = ['images', 'cropped', 'raw']
        images_dir = None

        for dir_name in possible_dirs:
            candidate = os.path.join(data_root, dir_name)
            if os.path.exists(candidate):
                images_dir = candidate
                print(f"[Info] Found data directory: {images_dir}")
                break

        if images_dir is None:
            print(f"[Warning] No data directory found in {data_root}")
            return None

        # 遍历subject/video结构
        for subject in os.listdir(images_dir)[:10]:  # 限制10个subject
            subject_dir = os.path.join(images_dir, subject)
            if not os.path.isdir(subject_dir):
                continue

            for video in os.listdir(subject_dir)[:5]:  # 每个subject限制5个video
                video_dir = os.path.join(images_dir, subject, video)
                if not os.path.isdir(video_dir):
                    continue

                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])

                if len(frames) >= self.min_frames:
                    # 随机分配情感（因为无标注）
                    emotion_idx = np.random.randint(0, 4)
                    emotion = self.emotion_filter[emotion_idx]

                    # 根据情感分配AU
                    au_dict = self._emotion_to_mock_au(emotion)

                    annotations.append({
                        'subject': subject,
                        'video': video,
                        'emotion': emotion,
                        'onset': 0,
                        'apex': len(frames) // 2,
                        'offset': len(frames) - 1,
                        'au_annotations': au_dict,
                        'num_frames': len(frames),
                        'video_dir': video_dir,
                        'frames': frames,
                    })

        return annotations

    def _emotion_to_mock_au(self, emotion):
        """情感到模拟AU的映射"""
        au_dict = {}

        if emotion == 'happiness':
            au_dict[AU_INDEX['AU6']] = 0.6 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU12']] = 0.7 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU25']] = 0.2 + np.random.rand() * 0.1

        elif emotion == 'surprise':
            au_dict[AU_INDEX['AU1']] = 0.6 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU2']] = 0.6 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU5']] = 0.7 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU25']] = 0.5 + np.random.rand() * 0.2

        elif emotion == 'disgust':
            au_dict[AU_INDEX['AU4']] = 0.4 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU9']] = 0.6 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU10']] = 0.4 + np.random.rand() * 0.1
            au_dict[AU_INDEX['AU17']] = 0.3 + np.random.rand() * 0.1

        elif emotion == 'repression':
            au_dict[AU_INDEX['AU14']] = 0.5 + np.random.rand() * 0.2
            au_dict[AU_INDEX['AU17']] = 0.4 + np.random.rand() * 0.1
            au_dict[AU_INDEX['AU4']] = 0.3 + np.random.rand() * 0.1

        return au_dict

    def _filter_samples(self):
        """过滤有效样本"""
        valid_samples = []

        if self.annotations is None:
            return valid_samples

        for ann in self.annotations:
            # 情感过滤
            if ann['emotion'] not in self.emotion_filter:
                continue

            # 帧数过滤
            if ann['num_frames'] < self.min_frames:
                continue

            # 检查视频目录是否存在
            if 'video_dir' not in ann:
                video_dir = os.path.join(
                    self.data_root, 'images',
                    ann['subject'], ann['video']
                )
                if os.path.exists(video_dir):
                    ann['video_dir'] = video_dir
                else:
                    continue

            valid_samples.append(ann)

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            dict:
                - neutral_face: (C, H, W) 中性脸
                - target_video: (C, T, H, W) 目标视频
                - au_activation: (17,) AU激活向量
                - emotion_class: 情感类别索引
                - intensity: 微表情强度
        """
        sample = self.samples[idx]

        # 加载帧图像
        frames = []
        video_dir = sample.get('video_dir', '')

        if 'frames' in sample:
            frame_files = sample['frames']
        else:
            frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])

        # 加载起始帧作为中性脸
        neutral_frame_path = os.path.join(video_dir, frame_files[0])
        neutral_face = self._load_image(neutral_frame_path)

        # 加载目标视频帧
        # 选择onset到offset范围内的帧
        onset = sample['onset']
        offset = sample['offset']

        target_frames = frame_files[onset:offset+1]

        for frame_file in target_frames:
            frame_path = os.path.join(video_dir, frame_file)
            frame = self._load_image(frame_path)
            frames.append(frame)

        # 填充或截断到num_frames
        frames = self._adjust_frames(frames)

        # 组合为视频张量
        target_video = torch.stack(frames, dim=1)  # (C, T, H, W)

        # 构建AU激活向量
        au_activation = torch.zeros(17)
        for au_idx, intensity in sample['au_annotations'].items():
            au_activation[au_idx] = intensity

        # 情感类别
        emotion_name = sample['emotion']
        emotion_class = CASME2_EMOTION_MAPPING.get(emotion_name, 0)

        # 微表情强度（估算）
        intensity = au_activation.max().item()

        return {
            'neutral_face': neutral_face,
            'target_video': target_video,
            'au_activation': au_activation,
            'emotion_class': emotion_class,
            'emotion_name': emotion_name,
            'intensity': intensity,
            'subject': sample['subject'],
            'video': sample['video'],
        }

    def _load_image(self, image_path):
        """加载并预处理图像"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            # 返回空白图像
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        img = cv2.resize(img, (self.image_size, self.image_size))

        # 转为张量
        img = torch.from_numpy(img).float().permute(2, 0, 1)  # (C, H, W)

        # 归一化
        if self.normalize:
            img = img / 255.0

        return img

    def _adjust_frames(self, frames):
        """调整帧数到num_frames"""
        T = len(frames)
        target_T = self.num_frames

        if T < target_T:
            # 填充：复制最后一帧
            last_frame = frames[-1]
            padding = [last_frame.clone() for _ in range(target_T - T)]
            frames = frames + padding

        elif T > target_T:
            # 截断：均匀采样
            indices = np.linspace(0, T-1, target_T).astype(int)
            frames = [frames[i] for i in indices]

        return frames


def create_casme2_dataloader(data_root: str,
                              batch_size: int = 8,
                              image_size: int = 224,
                              num_frames: int = 16,
                              shuffle: bool = True):
    """
    创建CASME2 DataLoader

    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        image_size: 图像尺寸
        num_frames: 视频帧数
        shuffle: 是否shuffle

    Returns:
        DataLoader
    """
    dataset = CASME2RealDataset(
        data_root=data_root,
        image_size=image_size,
        num_frames=num_frames,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


# =============================================================================
# Demo
# =============================================================================

def demo_casme2_loader():
    """Demo CASME2数据加载"""
    print("\n" + "="*60)
    print("CASME2 Real Dataset Demo")
    print("="*60)

    # 测试路径
    data_root = '/root/autodl-tmp/data/CASME2'  # AutoDL默认路径

    if not os.path.exists(data_root):
        print("\n[Warning] CASME2 not found, using mock data")
        # 使用模拟数据
        dataset = CASME2GeneratorDataset(
            data_root=None,
            num_samples=10,
            image_size=224,
            num_frames=16
        )
    else:
        dataset = CASME2RealDataset(
            data_root=data_root,
            image_size=224,
            num_frames=16,
        )

    if len(dataset) == 0:
        print("[Error] No valid samples found")
        return

    print(f"\n[Dataset Info]")
    print(f"  Total samples: {len(dataset)}")

    # 测试加载
    sample = dataset[0]
    print(f"\n[Sample 0]")
    print(f"  Neutral face shape: {sample['neutral_face'].shape}")
    print(f"  Target video shape: {sample['target_video'].shape}")
    print(f"  AU activation shape: {sample['au_activation'].shape}")
    print(f"  Emotion: {sample['emotion_name']} (class {sample['emotion_class']})")
    print(f"  Intensity: {sample['intensity']:.2f}")
    print(f"  Subject: {sample['subject']}, Video: {sample['video']}")

    # AU统计
    active_au = (sample['au_activation'] > 0.3).sum().item()
    print(f"  Active AU count: {active_au}")

    print("\n[Demo Complete]")


# =============================================================================
# Generator Dataset (Simulated for testing when no real data available)
# =============================================================================

class CASME2GeneratorDataset(Dataset):
    """
    模拟CASME2生成数据集（当真实数据不存在时使用）

    用于测试生成网络：
      - neutral_face: 模拟中性脸（随机噪声）
      - target_video: 模拟目标视频（随机噪声）
      - au_activation: 模拟AU激活
    """

    def __init__(self, data_root=None, num_samples=100, image_size=224, num_frames=16):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_frames = num_frames

        # 情感类别
        self.emotions = ['happiness', 'surprise', 'disgust', 'repression']

        print(f"[CASME2GeneratorDataset] Creating {num_samples} simulated samples")
        print(f"  Image size: {image_size}")
        print(f"  Num frames: {num_frames}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟中性脸
        neutral_face = torch.randn(3, self.image_size, self.image_size) * 0.3 + 0.5
        neutral_face = torch.clamp(neutral_face, 0, 1)

        # 模拟目标视频
        target_video = torch.randn(3, self.num_frames, self.image_size, self.image_size) * 0.3 + 0.5
        target_video = torch.clamp(target_video, 0, 1)

        # 模拟AU激活
        emotion_idx = idx % 4
        au_activation = torch.zeros(17)

        if emotion_idx == 0:  # happiness
            au_activation[AU_INDEX['AU6']] = 0.7
            au_activation[AU_INDEX['AU12']] = 0.8
        elif emotion_idx == 1:  # surprise
            au_activation[AU_INDEX['AU1']] = 0.6
            au_activation[AU_INDEX['AU2']] = 0.6
            au_activation[AU_INDEX['AU5']] = 0.7
        elif emotion_idx == 2:  # disgust
            au_activation[AU_INDEX['AU4']] = 0.5
            au_activation[AU_INDEX['AU9']] = 0.6
            au_activation[AU_INDEX['AU10']] = 0.4
        else:  # repression
            au_activation[AU_INDEX['AU14']] = 0.5
            au_activation[AU_INDEX['AU17']] = 0.4

        return {
            'neutral_face': neutral_face,
            'target_video': target_video,
            'au_activation': au_activation,
            'emotion_class': emotion_idx,
            'emotion_name': self.emotions[emotion_idx],
            'intensity': au_activation.max().item(),
            'subject': f'sim_{idx}',
            'video': f'video_{idx}',
        }


if __name__ == '__main__':
    demo_casme2_loader()


if __name__ == '__main__':
    demo_casme2_loader()