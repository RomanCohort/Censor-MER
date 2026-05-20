# =============================================================================
# Censor -- Ocular Motion Filter Module
# =============================================================================
# Filters out non-micro-expression ocular movements:
#   1. BlinkDetector: Detect and remove blink (AU45) interference
#   2. SaccadeDetector: Detect and mask rapid eye movements
#   3. OcularMotionFilter: Combined filtering pipeline
#   4. CleanSignalExtractor: Extract clean AU and flow signals
#
# Biological basis:
#   - Blink: AU45, periodic, ~300-400ms duration, ~15-20 times/min
#   - Saccade: Rapid eye movement, 30-100ms, velocity > 50 deg/s
#   - Smooth pursuit: Slow tracking movement, < 30 deg/s
#   - Gaze drift: Micro-drift during fixation, low frequency
#
# These movements create noise in:
#   - Optical flow (false motion signals)
#   - AU sequences (AU45 blink ≠ AU1+AU5 surprise)
#   - Temporal patterns (non-ME temporal dynamics)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.defaults import OCULAR_FILTER_CONFIG


# =============================================================================
# Blink Detection Constants
# =============================================================================
# Blink temporal characteristics (in frames, assuming 200fps video)

BLINK_CONFIG = {
    'min_duration_frames': 3,      # 最短眨眼帧数 (~15ms at 200fps)
    'max_duration_frames': 80,     # 最长眨眼帧数 (~400ms at 200fps)
    'typical_duration_frames': 60, #典型眨眼帧数 (~300ms at 200fps)
    'au45_threshold': 0.5,         # AU45激活阈值
    'velocity_threshold': 0.3,     # 速度阈值
    'min_interval_frames': 120,    # 最小眨眼间隔 (~600ms, normal ~15/min)
}


class BlinkDetector(nn.Module):
    """
    Detect blink patterns (AU45) in AU sequences.

    Blinks are periodic, predictable movements that should be filtered
    out from micro-expression analysis. They can be misclassified as
    surprise (AU1+AU5 combination) if not properly handled.

    Detection criteria:
      1. AU45 activation above threshold
      2. Duration within normal blink range (300-400ms)
      3. Temporal pattern: quick rise -> plateau -> quick fall
      4. Periodicity (normal ~15-20 blinks per minute)

    Architecture:
        AU45 sequence (B, T)
          -> Temporal Conv1d -> pattern detection
          -> Blink mask (B, T) where blinks occur
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or OCULAR_FILTER_CONFIG

        self.au45_threshold = cfg.get('blink_au45_threshold', BLINK_CONFIG['au45_threshold'])
        self.min_duration = cfg.get('blink_min_duration', BLINK_CONFIG['min_duration_frames'])
        self.max_duration = cfg.get('blink_max_duration', BLINK_CONFIG['max_duration_frames'])

        # Blink pattern detector (temporal convolution)
        # Detects the rise-plateau-fall pattern of blinks
        self.pattern_detector = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Duration classifier (is the duration typical for blink?)
        self.duration_classifier = nn.Sequential(
            nn.Linear(2, 16),  # (duration, peak_value)
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.pattern_detector[0].weight)
        nn.init.xavier_uniform_(self.duration_classifier[0].weight)

    def forward(self, au45_sequence, return_stats=False):
        """
        Args:
            au45_sequence (torch.Tensor): AU45 intensity sequence (B, T)
            return_stats (bool): Return blink statistics
        Returns:
            blink_mask (torch.Tensor): (B, T) mask where True indicates blink frames
            blink_stats (dict, optional): Statistics about detected blinks
        """
        B, T = au45_sequence.shape
        device = au45_sequence.device

        # Step 1: Threshold detection
        threshold_mask = au45_sequence > self.au45_threshold

        # Step 2: Pattern detection
        pattern_input = au45_sequence.unsqueeze(1)  # (B, 1, T)
        pattern_output = self.pattern_detector(pattern_input).squeeze(1)  # (B, T)

        # Step 3: Combine threshold and pattern
        blink_mask = threshold_mask & (pattern_output > 0.5)

        # Step 4: Validate duration (filter out very short spikes)
        blink_mask = self._validate_duration(blink_mask)

        # Compute statistics
        if return_stats:
            stats = self._compute_blink_stats(blink_mask, au45_sequence)
            return blink_mask, stats

        return blink_mask

    def _validate_duration(self, blink_mask):
        """
        Validate blink duration - filter out too short or too long events.

        Args:
            blink_mask (torch.Tensor): (B, T) raw mask
        Returns:
            validated_mask (torch.Tensor): (B, T) validated mask
        """
        B, T = blink_mask.shape
        validated_mask = torch.zeros_like(blink_mask)

        for b in range(B):
            # Find blink event start and end indices
            blink_events = self._find_events(blink_mask[b])

            for start, end in blink_events:
                duration = end - start

                # Check if duration is within valid range
                if self.min_duration <= duration <= self.max_duration:
                    validated_mask[b, start:end] = True

        return validated_mask

    def _find_events(self, mask):
        """
        Find contiguous True regions in mask.

        Args:
            mask (torch.Tensor): (T,) boolean mask
        Returns:
            events (list): List of (start, end) indices
        """
        events = []

        # Convert to numpy for easier processing
        mask_np = mask.cpu().numpy()
        diff = np.diff(mask_np.astype(int))

        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if mask_np[0]:
            starts = np.concatenate([[0], starts])
        if mask_np[-1]:
            ends = np.concatenate([ends, [len(mask_np)]])

        for start, end in zip(starts, ends):
            events.append((int(start), int(end)))

        return events

    def _compute_blink_stats(self, blink_mask, au45_sequence):
        """
        Compute statistics about detected blinks.

        Returns:
            stats (dict): {
                'num_blinks': int,
                'avg_duration': float,
                'blink_rate': float (blinks/min),
                'peak_intensity': float,
                'intervals': list
            }
        """
        B = blink_mask.shape[0]

        # Aggregate statistics across batch
        total_blinks = 0
        total_duration = 0
        peaks = []
        intervals = []

        for b in range(B):
            events = self._find_events(blink_mask[b])
            total_blinks += len(events)

            for start, end in events:
                duration = end - start
                total_duration += duration
                peak = au45_sequence[b, start:end].max().item()
                peaks.append(peak)

            # Compute intervals between blinks
            if len(events) > 1:
                for i in range(len(events) - 1):
                    interval = events[i+1][0] - events[i][1]
                    intervals.append(interval)

        # Compute averages
        avg_duration = total_duration / total_blinks if total_blinks > 0 else 0
        avg_peak = np.mean(peaks) if peaks else 0

        # Blink rate (assuming T frames at 200fps)
        # blink_rate = num_blinks / (T / fps / 60) = blinks per minute
        T = blink_mask.shape[1]
        fps = 200  # Typical MER dataset fps
        blink_rate = total_blinks / (T / fps / 60) if B > 0 else 0

        return {
            'num_blinks': total_blinks // B,  # Average per sample
            'avg_duration': avg_duration,
            'blink_rate': blink_rate,
            'peak_intensity': avg_peak,
            'intervals': intervals[:10] if intervals else []  # First 10
        }


class SaccadeDetector(nn.Module):
    """
    Detect rapid eye movements (saccades) from gaze sequence.

    Saccades are fast, ballistic eye movements (~30-100ms) that can
    create false motion signals in optical flow. They need to be
    filtered out to focus on facial muscle movements.

    Detection criteria:
      1. Velocity > threshold (typically > 50 deg/s)
      2. Duration < saccade max (typically < 100ms)
      3. Sharp velocity spike (peak-shaped)

    Architecture:
        Gaze sequence (B, T, 2)
          -> Velocity computation -> threshold detection
          -> Saccade mask (B, T)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or OCULAR_FILTER_CONFIG

        # Velocity threshold (in normalized units, ~50 deg/s equivalent)
        self.velocity_threshold = cfg.get('saccade_velocity_threshold', 0.5)
        self.max_duration = cfg.get('saccade_max_duration', 20)  # frames (~100ms at 200fps)

        # Acceleration threshold (for detecting saccade onset)
        self.accel_threshold = cfg.get('saccade_accel_threshold', 1.0)

        # Smoothing kernel for velocity computation
        self.smooth_kernel_size = cfg.get('velocity_smooth_kernel', 3)

    def forward(self, gaze_sequence, return_velocity=False):
        """
        Args:
            gaze_sequence (torch.Tensor): (B, T, 2) gaze positions over time
            return_velocity (bool): Return computed velocity
        Returns:
            saccade_mask (torch.Tensor): (B, T) mask where True indicates saccade frames
            velocity (torch.Tensor, optional): (B, T) gaze velocity
        """
        B, T, _ = gaze_sequence.shape
        device = gaze_sequence.device

        # Step 1: Compute gaze velocity (first derivative)
        velocity = self._compute_velocity(gaze_sequence)

        # Step 2: Smooth velocity to reduce noise
        velocity_smoothed = self._smooth_velocity(velocity)

        # Step 3: Threshold detection
        saccade_mask = velocity_smoothed > self.velocity_threshold

        # Step 4: Validate duration (filter out sustained movements)
        saccade_mask = self._validate_saccade_duration(saccade_mask)

        if return_velocity:
            return saccade_mask, velocity_smoothed

        return saccade_mask

    def _compute_velocity(self, gaze_sequence):
        """
        Compute gaze velocity from position sequence.

        Args:
            gaze_sequence (torch.Tensor): (B, T, 2)
        Returns:
            velocity (torch.Tensor): (B, T) magnitude of velocity
        """
        # First derivative: velocity = dx/dt
        # Use central difference for accuracy
        gaze_diff = torch.diff(gaze_sequence, dim=1)  # (B, T-1, 2)

        # Compute velocity magnitude
        velocity = torch.norm(gaze_diff, dim=-1)  # (B, T-1)

        # Pad to match original T
        velocity = F.pad(velocity.unsqueeze(1), (0, 1), mode='replicate').squeeze(1)  # (B, T)

        return velocity

    def _smooth_velocity(self, velocity):
        """
        Apply smoothing to velocity signal.

        Args:
            velocity (torch.Tensor): (B, T)
        Returns:
            smoothed (torch.Tensor): (B, T)
        """
        B, T = velocity.shape

        # Simple moving average smoothing
        kernel = torch.ones(1, 1, self.smooth_kernel_size) / self.smooth_kernel_size
        kernel = kernel.to(velocity.device)

        velocity_input = velocity.unsqueeze(1)  # (B, 1, T)
        smoothed = F.conv1d(velocity_input, kernel, padding=self.smooth_kernel_size // 2)
        smoothed = smoothed.squeeze(1)  # (B, T)

        return smoothed

    def _validate_saccade_duration(self, saccade_mask):
        """
        Validate saccade duration - filter out sustained movements.

        Saccades are brief (< 100ms), so filter out longer movements.
        """
        B, T = saccade_mask.shape
        validated_mask = torch.zeros_like(saccade_mask)

        for b in range(B):
            events = self._find_contiguous_events(saccade_mask[b])

            for start, end in events:
                duration = end - start

                # Saccades should be brief
                if duration <= self.max_duration:
                    validated_mask[b, start:end] = True

        return validated_mask

    def _find_contiguous_events(self, mask):
        """Find contiguous True regions."""
        events = []
        mask_np = mask.cpu().numpy()
        diff = np.diff(mask_np.astype(int))

        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if mask_np[0]:
            starts = np.concatenate([[0], starts])
        if mask_np[-1]:
            ends = np.concatenate([ends, [len(mask_np)]])

        for start, end in zip(starts, ends):
            events.append((int(start), int(end)))

        return events


class SmoothPursuitDetector(nn.Module):
    """
    Detect smooth pursuit eye movements (slow tracking).

    Smooth pursuit is slower (< 30 deg/s) and continuous, different
    from saccades. It's less interfering with ME detection but still
    should be tracked for comprehensive motion analysis.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or OCULAR_FILTER_CONFIG

        self.velocity_min = cfg.get('pursuit_velocity_min', 0.1)
        self.velocity_max = cfg.get('pursuit_velocity_max', 0.3)
        self.min_duration = cfg.get('pursuit_min_duration', 10)  # frames

    def forward(self, gaze_sequence, velocity=None):
        """
        Args:
            gaze_sequence (torch.Tensor): (B, T, 2)
            velocity (torch.Tensor, optional): Precomputed velocity (B, T)
        Returns:
            pursuit_mask (torch.Tensor): (B, T) smooth pursuit frames
        """
        if velocity is None:
            velocity = self._compute_velocity(gaze_sequence)

        # Smooth pursuit: moderate velocity, sustained
        pursuit_mask = (velocity > self.velocity_min) & (velocity < self.velocity_max)

        # Validate duration (smooth pursuit is sustained)
        pursuit_mask = self._validate_duration(pursuit_mask)

        return pursuit_mask

    def _compute_velocity(self, gaze_sequence):
        """Compute gaze velocity."""
        gaze_diff = torch.diff(gaze_sequence, dim=1)
        velocity = torch.norm(gaze_diff, dim=-1)
        velocity = F.pad(velocity.unsqueeze(1), (0, 1), mode='replicate').squeeze(1)
        return velocity

    def _validate_duration(self, pursuit_mask):
        """Filter out brief movements."""
        B, T = pursuit_mask.shape
        validated = torch.zeros_like(pursuit_mask)

        for b in range(B):
            mask_np = pursuit_mask[b].cpu().numpy()
            diff = np.diff(mask_np.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1

            if mask_np[0]:
                starts = np.concatenate([[0], starts])
            if mask_np[-1]:
                ends = np.concatenate([ends, [T]])

            for start, end in zip(starts, ends):
                if end - start >= self.min_duration:
                    validated[b, start:end] = True

        return validated


class OcularMotionFilter(nn.Module):
    """
    Combined ocular motion filtering pipeline.

    Filters multiple types of ocular interference:
      1. Blink (AU45) - from AU sequences
      2. Saccade - from optical flow and gaze
      3. Smooth pursuit - from gaze (optional)

    Provides clean signals for micro-expression analysis.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or OCULAR_FILTER_CONFIG

        # Individual detectors
        self.blink_detector = BlinkDetector(cfg)
        self.saccade_detector = SaccadeDetector(cfg)
        self.pursuit_detector = SmoothPursuitDetector(cfg)

        # Interference mask combiner
        self.combination_mode = cfg.get('combination_mode', 'union')

        # Filter strength (0-1, how much to suppress)
        self.filter_strength = cfg.get('filter_strength', 0.8)

        # AU remapping (AU45 → 0 after filtering)
        self.au45_index = 45 - 1  # AU indexing (AU1=0, AU45=44)

    def forward(self, au_sequence, optical_flow, gaze_sequence=None,
                return_masks=False, return_stats=False):
        """
        Args:
            au_sequence (torch.Tensor): (B, T, num_aus) AU intensities
            optical_flow (torch.Tensor): (B, 2, T, H, W) optical flow
            gaze_sequence (torch.Tensor, optional): (B, T, 2) gaze positions
            return_masks (bool): Return individual interference masks
            return_stats (bool): Return statistics
        Returns:
            clean_au (torch.Tensor): (B, T, num_aus) filtered AU sequence
            clean_flow (torch.Tensor): (B, 2, T, H, W) filtered optical flow
            masks (dict, optional): Individual interference masks
            stats (dict, optional): Filtering statistics
        """
        B, T = au_sequence.shape[:2]
        device = au_sequence.device

        # Step 1: Detect blinks
        au45_sequence = au_sequence[:, :, self.au45_index] if au_sequence.shape[2] > self.au45_index else au_sequence[:, :, 0]
        blink_mask = self.blink_detector(au45_sequence)
        blink_stats = None
        if return_stats:
            blink_mask, blink_stats = self.blink_detector(au45_sequence, return_stats=True)

        # Step 2: Detect saccades (if gaze available)
        saccade_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
        velocity = None
        if gaze_sequence is not None:
            saccade_mask, velocity = self.saccade_detector(gaze_sequence, return_velocity=True)

        # Step 3: Detect smooth pursuit (optional)
        pursuit_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
        if gaze_sequence is not None and velocity is not None:
            pursuit_mask = self.pursuit_detector(gaze_sequence, velocity)

        # Step 4: Combine interference masks
        if self.combination_mode == 'union':
            interference_mask = blink_mask | saccade_mask | pursuit_mask
        elif self.combination_mode == 'blink_only':
            interference_mask = blink_mask
        elif self.combination_mode == 'weighted':
            # Weighted combination (blink most important)
            interference_float = (
                0.5 * blink_mask.float() +
                0.3 * saccade_mask.float() +
                0.2 * pursuit_mask.float()
            )
            interference_mask = interference_float > 0.3
        else:
            interference_mask = blink_mask | saccade_mask

        # Step 5: Apply filtering
        clean_au = self._filter_au_sequence(au_sequence, blink_mask, interference_mask)
        clean_flow = self._filter_optical_flow(optical_flow, interference_mask)

        # Prepare outputs
        outputs = {'clean_au': clean_au, 'clean_flow': clean_flow}

        if return_masks:
            outputs['masks'] = {
                'blink': blink_mask,
                'saccade': saccade_mask,
                'pursuit': pursuit_mask,
                'interference': interference_mask
            }

        if return_stats:
            outputs['stats'] = {
                'blink': blink_stats,
                'saccade': self._compute_saccade_stats(saccade_mask),
                'filter_ratio': interference_mask.float().mean().item()
            }

        return outputs

    def _filter_au_sequence(self, au_sequence, blink_mask, interference_mask):
        """
        Filter AU sequence by suppressing interference frames.

        Special handling for AU45 (blink) - set to 0 or interpolate.
        """
        B, T, num_aus = au_sequence.shape
        clean_au = au_sequence.clone()

        # Create suppression factor (0 at interference, 1 elsewhere)
        suppression = (~interference_mask).float().unsqueeze(-1)  # (B, T, 1)

        # Apply suppression with filter_strength
        clean_au = clean_au * suppression + clean_au * (1 - self.filter_strength) * interference_mask.unsqueeze(-1).float()

        # Special handling: Set AU45 to 0 during detected blinks
        # (The blink itself is not a micro-expression AU)
        if au_sequence.shape[2] > self.au45_index:
            au45_clean = clean_au[:, :, self.au45_index].clone()
            au45_clean[blink_mask] = 0
            clean_au[:, :, self.au45_index] = au45_clean

        return clean_au

    def _filter_optical_flow(self, optical_flow, interference_mask):
        """
        Filter optical flow by masking interference frames.

        Optical flow during blinks/saccades is mostly eye movement,
        not facial muscle movement, so it should be suppressed.
        """
        B, C, T, H, W = optical_flow.shape
        clean_flow = optical_flow.clone()

        # Create temporal mask (B, 1, T, 1, 1) for broadcasting
        temporal_mask = (~interference_mask).float().unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # Apply suppression
        clean_flow = clean_flow * temporal_mask + clean_flow * (1 - self.filter_strength) * (1 - temporal_mask)

        return clean_flow

    def _compute_saccade_stats(self, saccade_mask):
        """Compute saccade statistics."""
        B, T = saccade_mask.shape

        total_saccades = 0
        total_duration = 0

        for b in range(B):
            mask_np = saccade_mask[b].cpu().numpy()
            diff = np.diff(mask_np.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1

            if mask_np[0]:
                starts = np.concatenate([[0], starts])
            if mask_np[-1]:
                ends = np.concatenate([ends, [T]])

            total_saccades += len(starts)
            for start, end in zip(starts, ends):
                total_duration += (end - start)

        return {
            'num_saccades': total_saccades // B if B > 0 else 0,
            'avg_duration': total_duration / total_saccades if total_saccades > 0 else 0
        }


class CleanSignalExtractor(nn.Module):
    """
    Extract clean micro-expression signals from noisy input.

    Combines OcularMotionFilter with additional signal enhancement:
      1. Temporal smoothing
      2. Baseline normalization
      3. Signal quality scoring
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or OCULAR_FILTER_CONFIG

        self.ocular_filter = OcularMotionFilter(cfg)

        # Temporal smoothing
        self.smooth_window = cfg.get('signal_smooth_window', 3)

        # Baseline normalization
        self.baseline_frames = cfg.get('baseline_frames', 3)

        # Quality scorer (estimates signal quality after filtering)
        self.quality_scorer = nn.Sequential(
            nn.Linear(4, 16),  # (blink_ratio, saccade_ratio, smooth_ratio, filter_ratio)
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, au_sequence, optical_flow, gaze_sequence=None,
                return_quality=True):
        """
        Args:
            au_sequence (torch.Tensor): (B, T, num_aus)
            optical_flow (torch.Tensor): (B, 2, T, H, W)
            gaze_sequence (torch.Tensor, optional): (B, T, 2)
            return_quality (bool): Return signal quality score
        Returns:
            clean_signals (dict): {
                'au': clean AU sequence,
                'flow': clean optical flow,
                'baseline': baseline AU values,
                'quality': signal quality score (optional)
            }
        """
        # Apply ocular motion filter
        filter_outputs = self.ocular_filter(
            au_sequence, optical_flow, gaze_sequence,
            return_masks=True, return_stats=True
        )

        clean_au = filter_outputs['clean_au']
        clean_flow = filter_outputs['clean_flow']

        # Apply temporal smoothing
        clean_au = self._temporal_smooth(clean_au)
        clean_flow = self._temporal_smooth_flow(clean_flow)

        # Compute baseline (resting AU values)
        baseline = self._compute_baseline(clean_au)

        # Compute signal quality
        stats = filter_outputs['stats']
        quality = None
        if return_quality:
            quality_input = torch.tensor([
                stats['blink'].get('num_blinks', 0) / au_sequence.shape[1],  # blink_ratio
                stats['saccade'].get('num_saccades', 0) / au_sequence.shape[1],  # saccade_ratio
                0,  # pursuit_ratio placeholder
                stats['filter_ratio']
            ], device=au_sequence.device).unsqueeze(0).expand(au_sequence.shape[0], -1)
            quality = self.quality_scorer(quality_input)

        return {
            'au': clean_au,
            'flow': clean_flow,
            'baseline': baseline,
            'masks': filter_outputs['masks'],
            'stats': stats,
            'quality': quality
        }

    def _temporal_smooth(self, au_sequence):
        """Apply temporal smoothing to AU sequence."""
        B, T, num_aus = au_sequence.shape

        # Moving average smoothing
        kernel = torch.ones(1, 1, self.smooth_window) / self.smooth_window
        kernel = kernel.to(au_sequence.device)

        smoothed = []
        for au_idx in range(num_aus):
            au_channel = au_sequence[:, :, au_idx].unsqueeze(1)  # (B, 1, T)
            smoothed_channel = F.conv1d(au_channel, kernel, padding=self.smooth_window // 2)
            smoothed.append(smoothed_channel.squeeze(1))

        smoothed_au = torch.stack(smoothed, dim=-1)  # (B, T, num_aus)
        return smoothed_au

    def _temporal_smooth_flow(self, optical_flow):
        """Apply temporal smoothing to optical flow."""
        B, C, T, H, W = optical_flow.shape

        # Average pooling over time (simplified smoothing)
        smoothed = F.avg_pool3d(
            optical_flow,
            kernel_size=(self.smooth_window, 1, 1),
            stride=1,
            padding=(self.smooth_window // 2, 0, 0)
        )

        return smoothed

    def _compute_baseline(self, au_sequence):
        """
        Compute baseline AU values from first few frames.

        Baseline represents the person's resting expression.
        """
        B, T, num_aus = au_sequence.shape

        # Use first baseline_frames frames for baseline
        baseline = au_sequence[:, :self.baseline_frames].mean(dim=1)  # (B, num_aus)

        return baseline


# =============================================================================
# Factory Functions
# =============================================================================

def create_ocular_filter(config=None):
    """Factory function to create OcularMotionFilter."""
    return OcularMotionFilter(config or OCULAR_FILTER_CONFIG)


def create_clean_signal_extractor(config=None):
    """Factory function to create CleanSignalExtractor."""
    return CleanSignalExtractor(config or OCULAR_FILTER_CONFIG)


def create_blink_detector(config=None):
    """Factory function to create BlinkDetector."""
    return BlinkDetector(config or OCULAR_FILTER_CONFIG)


def create_saccade_detector(config=None):
    """Factory function to create SaccadeDetector."""
    return SaccadeDetector(config or OCULAR_FILTER_CONFIG)