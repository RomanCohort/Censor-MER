# =============================================================================
# Censor -- CV Emotion Bridge Module
# =============================================================================
# Real-time emotion detection from camera/video frames.
#
# Features:
#   1. Real-time face detection + emotion classification
#   2. Multi-modal signals: facial expression + rPPG heart rate + eye gaze
#   3. Temporal smoothing for stable predictions
#   4. Integration with Censor MER pipeline
#   5. Civis Lucri-Faber compatible output format
#
# Backends:
#   - FER: Fast emotion detection (MTCNN + CNN)
#   - DeepFace: Multi-model ensemble
#   - Built-in: Use Censor's own MER model (recommended)
# =============================================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EmotionDetectionResult:
    """Single-frame emotion detection result."""
    # Core emotion
    emotion_tensor: torch.Tensor      # [7] emotion probabilities
    dominant_emotion: str             # 'happy', 'sad', etc.
    confidence: float                 # 0.0 - 1.0

    # Face metadata
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    landmarks: Optional[np.ndarray] = None            # 68 facial landmarks

    # Multi-modal signals
    rppg_heart_rate: Optional[float] = None           # BPM
    gaze_direction: Optional[Tuple[float, float]] = None  # (x, y) normalized
    blink_rate: Optional[float] = None                # blinks per minute

    # Temporal
    timestamp: float = 0.0

    # Raw features (for downstream integration)
    raw_features: Optional[torch.Tensor] = None       # [512] or [768]


@dataclass
class TemporalEmotionState:
    """Smoothed temporal emotion state."""
    # Smoothed emotion (over N frames)
    smoothed_tensor: torch.Tensor     # [7]
    smoothed_dominant: str
    smoothed_confidence: float

    # Dynamics
    emotion_velocity: float           # Rate of change
    criticality: float                # Near threshold?

    # History
    history_length: int

    # Meta
    fps: float
    latency_ms: float


# =============================================================================
# Backend Abstractions
# =============================================================================

class EmotionBackend:
    """Abstract backend for emotion detection."""

    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def detect(self, frame: np.ndarray) -> EmotionDetectionResult:
        raise NotImplementedError

    def detect_batch(self, frames: List[np.ndarray]) -> List[EmotionDetectionResult]:
        return [self.detect(f) for f in frames]

    def _to_tensor(self, emotions: Dict[str, float]) -> torch.Tensor:
        """Convert emotion dict to 7-dim tensor."""
        vals = [emotions.get(e, 0.0) for e in self.EMOTION_LABELS]
        return torch.tensor(vals, dtype=torch.float32)


class FERBackend(EmotionBackend):
    """
    FER (Facial Expression Recognition) backend.

    Uses MTCNN for face detection + CNN for emotion classification.
    Fast but less accurate than ensemble methods.

    pip install fer
    """

    def __init__(self, mtcnn: bool = True):
        self.available = False
        self.detector = None

        # Try multiple import paths (FER package structure changed over versions)
        try:
            # New FER package (v25+)
            from fer.fer import FER
            self.detector = FER(mtcnn=mtcnn)
            self.available = True
            logger.info("[FERBackend] Initialized (from fer.fer)")
        except ImportError:
            try:
                # Old FER package
                from fer import FER
                self.detector = FER(mtcnn=mtcnn)
                self.available = True
                logger.info("[FERBackend] Initialized (from fer)")
            except ImportError:
                logger.warning("[FERBackend] FER not installed. Run: pip install fer")
            except Exception as e:
                logger.warning(f"[FERBackend] FER initialization error: {e}")

    def detect(self, frame: np.ndarray) -> EmotionDetectionResult:
        if not self.available:
            return self._fallback()

        try:
            result = self.detector.detect_emotions(frame)

            if not result:
                return self._fallback()

            top_face = result[0]
            emotions = top_face['emotions']

            return EmotionDetectionResult(
                emotion_tensor=self._to_tensor(emotions),
                dominant_emotion=max(emotions, key=emotions.get),
                confidence=max(emotions.values()),
                bbox=top_face['box'],
                timestamp=time.time(),
            )
        except Exception as e:
            logger.error(f"[FERBackend] Detection error: {e}")
            return self._fallback()

    def _fallback(self) -> EmotionDetectionResult:
        return EmotionDetectionResult(
            emotion_tensor=torch.zeros(7),
            dominant_emotion='neutral',
            confidence=0.5,
            timestamp=time.time(),
        )


class DeepFaceBackend(EmotionBackend):
    """
    DeepFace backend with multi-model ensemble.

    Supports multiple backends: VGG-Face, Google FaceNet, OpenFace, DeepID, Dlib, ArcFace.
    More accurate but slower than FER.

    pip install deepface
    """

    def __init__(self, model_name: str = "Emotion"):
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.model_name = model_name
            self.available = True
            logger.info(f"[DeepFaceBackend] Initialized with {model_name}")
        except ImportError:
            self.available = False
            logger.warning("[DeepFaceBackend] DeepFace not installed. Run: pip install deepface")

    def detect(self, frame: np.ndarray) -> EmotionDetectionResult:
        if not self.available:
            return self._fallback()

        try:
            # DeepFace expects RGB, cv2 gives BGR
            result = self.deepface.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'neutral')

            # bbox
            region = result.get('region', {})
            bbox = (region.get('x', 0), region.get('y', 0),
                    region.get('w', 0), region.get('h', 0))

            return EmotionDetectionResult(
                emotion_tensor=self._to_tensor(emotions),
                dominant_emotion=dominant,
                confidence=max(emotions.values()) if emotions else 0.5,
                bbox=bbox,
                timestamp=time.time(),
            )
        except Exception as e:
            logger.error(f"[DeepFaceBackend] Detection error: {e}")
            return self._fallback()

    def _fallback(self) -> EmotionDetectionResult:
        return EmotionDetectionResult(
            emotion_tensor=torch.zeros(7),
            dominant_emotion='neutral',
            confidence=0.5,
            timestamp=time.time(),
        )


class CensorBackend(EmotionBackend):
    """
    Use Censor's own MER model for emotion detection.

    Most integrated option: uses the full biomimetic dual-pathway pipeline.
    Requires 16 frames of temporal context for optimal performance.
    """

    def __init__(self, censor_model: nn.Module = None, device: str = 'cpu'):
        self.model = censor_model
        self.device = device
        self.available = False  # 初始化时标记为不可用，lazy load后更新

        # Frame buffer for temporal input (Censor expects T=16)
        self.frame_buffer = deque(maxlen=16)

        # rPPG extractor (simple version)
        self._rppg_history = deque(maxlen=60)

        if self.model is not None:
            self.available = True
            logger.info(f"[CensorBackend] Initialized with Censor model on {device}")
        else:
            logger.warning("[CensorBackend] No model provided. Will try lazy loading.")

    def _load_model(self):
        """Lazy load Censor model if not provided."""
        if self.model is None:
            try:
                import sys
                from pathlib import Path
                censor_path = Path(__file__).parent.parent
                sys.path.insert(0, str(censor_path))

                from main import Censor
                self.model = Censor()
                self.model.eval()
                self.model.to(self.device)
                self.available = True
                logger.info("[CensorBackend] Model loaded successfully")
            except Exception as e:
                logger.error(f"[CensorBackend] Failed to load model: {e}")
                self.model = None
                self.available = False

    def detect(self, frame: np.ndarray) -> EmotionDetectionResult:
        """Single frame detection (uses buffered frames for temporal context)."""
        self._load_model()

        if self.model is None:
            return self._fallback(frame)

        # Preprocess frame
        frame_tensor = self._preprocess_frame(frame)
        self.frame_buffer.append(frame_tensor)

        # Need at least 4 frames for meaningful prediction
        if len(self.frame_buffer) < 4:
            return self._fallback(frame, partial=True)

        # Build temporal input
        video_tensor = self._build_temporal_input()

        try:
            with torch.no_grad():
                outputs = self.model(video_tensor)

            # Extract emotion logits
            me_logits = outputs['me_logits'][0]  # [7]
            emotion_probs = F.softmax(me_logits, dim=0)

            # AU intensities for additional features
            au_intensities = outputs['au_intensities'][0]  # [T, 28]
            au_avg = au_intensities.mean(dim=0)  # [28]

            # Expert gates (personalization info)
            expert_gates = outputs['expert_gates'][0]  # [3]

            # Apex timing
            apex_scores = outputs['apex_scores'][0]
            apex_idx = apex_scores.argmax().item() if len(apex_scores) > 0 else 0

            # rPPG estimate (simple pulse detection)
            rppg_hr = self._estimate_rppg(frame)

            return EmotionDetectionResult(
                emotion_tensor=emotion_probs,
                dominant_emotion=self.EMOTION_LABELS[emotion_probs.argmax().item()],
                confidence=emotion_probs.max().item(),
                bbox=None,  # Censor processes full frame
                rppg_heart_rate=rppg_hr,
                timestamp=time.time(),
                # Raw features for downstream integration
                raw_features=torch.cat([
                    emotion_probs,
                    au_avg[:7],  # Top 7 AUs
                    expert_gates,
                ]),  # [17]
            )
        except Exception as e:
            logger.error(f"[CensorBackend] Model inference error: {e}")
            return self._fallback(frame)

    def detect_sequence(self, frames: List[np.ndarray]) -> List[EmotionDetectionResult]:
        """Optimized batch processing for video sequences."""
        if len(frames) < 16:
            logger.warning("[CensorBackend] Sequence too short, using single-frame mode")
            return [self.detect(f) for f in frames]

        # Build video tensor directly
        video_tensor = torch.stack([self._preprocess_frame(f) for f in frames[:16]])
        video_tensor = video_tensor.unsqueeze(0).permute(0, 4, 1, 2, 3)  # [1, 3, 16, H, W]
        video_tensor = video_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(video_tensor)

        # Generate per-frame estimates from temporal output
        me_logits = outputs['me_logits'][0]
        emotion_probs = F.softmax(me_logits, dim=0)

        results = []
        for i, frame in enumerate(frames):
            results.append(EmotionDetectionResult(
                emotion_tensor=emotion_probs,
                dominant_emotion=self.EMOTION_LABELS[emotion_probs.argmax().item()],
                confidence=emotion_probs.max().item(),
                timestamp=time.time() + i * 0.033,  # ~30fps
                raw_features=emotion_probs,
            ))

        return results

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert cv2 frame to tensor for Censor."""
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 224x224
        resized = cv2.resize(rgb, (224, 224))
        # Normalize [0, 255] -> [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # To tensor [H, W, C]
        tensor = torch.from_numpy(normalized)
        return tensor

    def _build_temporal_input(self) -> torch.Tensor:
        """Build 16-frame video tensor from buffer."""
        frames = list(self.frame_buffer)

        # Pad if less than 16
        while len(frames) < 16:
            frames.append(frames[-1])  # Repeat last frame

        # Stack and reshape
        video = torch.stack(frames[-16:])  # [16, H, W, C]
        video = video.unsqueeze(0).permute(0, 4, 1, 2, 3)  # [1, 3, 16, 224, 224]
        return video.to(self.device)

    def _estimate_rppg(self, frame: np.ndarray) -> Optional[float]:
        """Simple rPPG heart rate estimation."""
        # Very simplified: track average green channel intensity
        green = frame[:, :, 1].mean()
        self._rppg_history.append(green)

        if len(self._rppg_history) < 30:
            return None

        # FFT-based pulse detection (simplified)
        signal = np.array(self._rppg_history)
        signal = signal - signal.mean()

        # Assume 30 fps, pulse range 0.75-2.5 Hz (45-150 BPM)
        fft = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(signal), d=1/30)

        # Find peak in valid range
        valid_mask = (freqs > 0.75) & (freqs < 2.5)
        if not valid_mask.any():
            return None

        valid_fft = fft[valid_mask]
        valid_freqs = freqs[valid_mask]

        peak_idx = valid_fft.argmax()
        peak_freq = valid_freqs[peak_idx]

        return peak_freq * 60  # Hz -> BPM

    def _fallback(self, frame: np.ndarray, partial: bool = False) -> EmotionDetectionResult:
        rppg_hr = self._estimate_rppg(frame)
        return EmotionDetectionResult(
            emotion_tensor=torch.ones(7) / 7 if partial else torch.zeros(7),
            dominant_emotion='neutral' if partial else 'unknown',
            confidence=0.3 if partial else 0.0,
            rppg_heart_rate=rppg_hr,
            timestamp=time.time(),
        )


# =============================================================================
# Multi-Modal Enhancement
# =============================================================================

class MultiModalEnhancer:
    """
    Enhance emotion detection with multi-modal signals.

    Signals:
      - rPPG: Heart rate variability -> stress/arousal
      - Eye gaze: Attention direction -> interest/avoidance
      - Blink rate: Cognitive load -> concentration/stress
      - Head pose: Engagement/disengagement
    """

    def __init__(self):
        self._eye_history = deque(maxlen=30)
        self._blink_counter = 0
        self._last_blink_time = time.time()

    def enhance(self, result: EmotionDetectionResult, frame: np.ndarray) -> EmotionDetectionResult:
        """Add multi-modal signals to detection result."""

        # 1. Eye gaze (simplified: track eye region movement)
        gaze = self._estimate_gaze(frame)
        if gaze is not None:
            result.gaze_direction = gaze

        # 2. Blink detection
        blink_rate = self._detect_blinks(frame)
        if blink_rate is not None:
            result.blink_rate = blink_rate

        return result

    def _estimate_gaze(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Estimate gaze direction from eye regions."""
        # Simplified: use face detection to find eye regions
        # In production, use proper eye tracking library
        try:
            # Use OpenCV's built-in face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return None

            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) < 2:
                return None

            # Estimate gaze from eye positions relative to face
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]

            # Normalized gaze direction
            gaze_x = (ex1 + ex2 + ew1 + ew2) / 2 / w - 0.5
            gaze_y = (ey1 + ey2 + eh1 + eh2) / 2 / h - 0.5

            return (float(gaze_x), float(gaze_y))
        except:
            return None

    def _detect_blinks(self, frame: np.ndarray) -> Optional[float]:
        """Detect blink rate from frame sequence."""
        # Simplified: track eye aspect ratio
        # In production, use proper blink detection
        self._eye_history.append(frame)

        if len(self._eye_history) < 30:
            return None

        # Count blinks per minute estimate
        # Placeholder: average human blink rate 15-20/min
        return 15.0


# =============================================================================
# Temporal Smoothing & Dynamics
# =============================================================================

class TemporalSmoother:
    """
    Temporal smoothing and dynamics tracking.

    Features:
      - Moving average smoothing
      - Exponential decay smoothing
      - Velocity (rate of change) tracking
      - Criticality detection (near-threshold states)
    """

    def __init__(self, window_size: int = 5, decay_rate: float = 0.9):
        self.window_size = window_size
        self.decay_rate = decay_rate

        self._history = deque(maxlen=window_size)
        self._smoothed = None
        self._prev_smoothed = None

    def update(self, result: EmotionDetectionResult) -> TemporalEmotionState:
        """Update smoothing and compute dynamics."""

        # Add to history
        self._history.append(result.emotion_tensor)

        # Compute smoothed
        if self._smoothed is None:
            self._smoothed = result.emotion_tensor.clone()
        else:
            # Exponential decay smoothing
            self._smoothed = self.decay_rate * self._smoothed + \
                            (1 - self.decay_rate) * result.emotion_tensor

        # Track previous for velocity
        if self._prev_smoothed is None:
            self._prev_smoothed = self._smoothed.clone()

        # Compute velocity
        velocity = torch.norm(self._smoothed - self._prev_smoothed).item()

        # Compute criticality (entropy-based)
        entropy = -torch.sum(self._smoothed * torch.log(self._smoothed + 1e-8))
        max_entropy = np.log(7)  # Maximum entropy for 7 emotions
        criticality = 1.0 - (entropy / max_entropy).item()  # High criticality = low entropy (uncertain)

        # Update previous
        self._prev_smoothed = self._smoothed.clone()

        # Dominant emotion from smoothed
        dominant_idx = self._smoothed.argmax().item()
        dominant = EmotionBackend.EMOTION_LABELS[dominant_idx]
        confidence = self._smoothed[dominant_idx].item()

        return TemporalEmotionState(
            smoothed_tensor=self._smoothed.clone(),
            smoothed_dominant=dominant,
            smoothed_confidence=confidence,
            emotion_velocity=velocity,
            criticality=criticality,
            history_length=len(self._history),
            fps=30.0,  # Approximate
            latency_ms=0.0,
        )

    def reset(self):
        """Reset smoothing state."""
        self._history.clear()
        self._smoothed = None
        self._prev_smoothed = None


# =============================================================================
# Main Bridge Class
# =============================================================================

class CVEmotionBridge:
    """
    CV Emotion Detection Bridge.

    Integrates multiple backends and provides:
      - Real-time emotion detection from camera/video
      - Multi-modal signal enhancement
      - Temporal smoothing and dynamics
      - Civis Lucri-Faber compatible output format
      - Easy integration with Censor MER pipeline

    Usage:
        bridge = CVEmotionBridge(backend='censor')

        # From camera
        for frame in camera_stream:
            result = bridge.detect(frame)
            print(f"Emotion: {result.dominant_emotion}, Confidence: {result.confidence}")

        # From video file
        results = bridge.process_video('video.mp4')

        # Civis integration
        emotion_tensor = bridge.get_civis_compatible_output()
    """

    BACKEND_OPTIONS = ['fer', 'deepface', 'censor', 'auto']

    def __init__(
        self,
        backend: str = 'auto',
        censor_model: nn.Module = None,
        device: str = 'cpu',
        smoothing_window: int = 5,
        enable_multimodal: bool = True,
    ):
        self.backend_name = backend
        self.device = device
        self.enable_multimodal = enable_multimodal

        # Initialize backend
        self.backend = self._init_backend(backend, censor_model, device)

        # Temporal smoothing
        self.smoother = TemporalSmoother(window_size=smoothing_window)

        # Multi-modal enhancement
        self.enhancer = MultiModalEnhancer() if enable_multimodal else None

        # Latest results
        self._latest_result: Optional[EmotionDetectionResult] = None
        self._latest_state: Optional[TemporalEmotionState] = None

        # Performance tracking
        self._frame_count = 0
        self._start_time = time.time()

        logger.info(f"[CVEmotionBridge] Initialized with backend={backend}, device={device}")

    def _init_backend(self, name: str, censor_model: nn.Module, device: str) -> EmotionBackend:
        """Initialize emotion detection backend."""
        if name == 'fer':
            return FERBackend(mtcnn=True)
        elif name == 'deepface':
            return DeepFaceBackend()
        elif name == 'censor':
            return CensorBackend(censor_model=censor_model, device=device)
        elif name == 'auto':
            # Try censor first (most integrated), then deepface, then fer
            if censor_model is not None:
                return CensorBackend(censor_model=censor_model, device=device)
            try:
                return DeepFaceBackend()
            except:
                try:
                    return FERBackend()
                except:
                    logger.warning("[CVEmotionBridge] No backend available, using fallback")
                    return FERBackend()
        else:
            logger.warning(f"[CVEmotionBridge] Unknown backend '{name}', using auto")
            return self._init_backend('auto', censor_model, device)

    def detect(self, frame: np.ndarray, smooth: bool = True) -> EmotionDetectionResult:
        """
        Detect emotion from single frame.

        Args:
            frame: BGR image (cv2 format), shape (H, W, 3)
            smooth: Apply temporal smoothing

        Returns:
            EmotionDetectionResult with emotion probabilities and metadata
        """
        # Raw detection
        result = self.backend.detect(frame)

        # Multi-modal enhancement
        if self.enhancer is not None:
            result = self.enhancer.enhance(result, frame)

        # Temporal smoothing
        if smooth:
            state = self.smoother.update(result)
            result.emotion_tensor = state.smoothed_tensor
            result.confidence = state.smoothed_confidence
            self._latest_state = state
        else:
            self.smoother.reset()

        # Store latest
        self._latest_result = result
        self._frame_count += 1

        return result

    def detect_sequence(self, frames: List[np.ndarray]) -> List[EmotionDetectionResult]:
        """
        Detect emotions from frame sequence.

        Optimized for video processing - uses batch inference when possible.
        """
        if isinstance(self.backend, CensorBackend):
            # Censor backend has optimized sequence processing
            results = self.backend.detect_sequence(frames)
        else:
            results = self.backend.detect_batch(frames)

        # Apply smoothing to sequence
        smoothed_results = []
        for r in results:
            if self.enhancer:
                r = self.enhancer.enhance(r, frames[results.index(r)])
            state = self.smoother.update(r)
            r.emotion_tensor = state.smoothed_tensor
            r.confidence = state.smoothed_confidence
            smoothed_results.append(r)

        return smoothed_results

    def process_video(self, video_path: str, fps: int = 30) -> List[EmotionDetectionResult]:
        """
        Process video file and return emotion timeline.

        Args:
            video_path: Path to video file
            fps: Processing frame rate (default 30)

        Returns:
            List of EmotionDetectionResult for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        results = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (30 // fps) == 0:  # Skip frames for lower fps
                result = self.detect(frame)
                results.append(result)

            frame_idx += 1

        cap.release()
        return results

    def run_camera(self, camera_id: int = 0, display: bool = True):
        """
        Run real-time detection from camera.

        Args:
            camera_id: Camera device ID (0 for default)
            display: Show detection overlay on frame
        """
        cap = cv2.VideoCapture(camera_id)

        logger.info(f"[CVEmotionBridge] Starting camera {camera_id}")
        logger.info("Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.detect(frame)

            if display:
                frame = self._overlay_result(frame, result)
                cv2.imshow('CV Emotion Bridge', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def _overlay_result(self, frame: np.ndarray, result: EmotionDetectionResult) -> np.ndarray:
        """Draw detection result overlay on frame."""
        # Emotion label
        emotion_text = f"{result.dominant_emotion}: {result.confidence:.2f}"
        cv2.putText(frame, emotion_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Face bbox
        if result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Heart rate
        if result.rppg_heart_rate is not None:
            hr_text = f"HR: {result.rppg_heart_rate:.0f} BPM"
            cv2.putText(frame, hr_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Gaze
        if result.gaze_direction is not None:
            gaze_text = f"Gaze: ({result.gaze_direction[0]:.2f}, {result.gaze_direction[1]:.2f})"
            cv2.putText(frame, gaze_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    # =========================================================================
    # Civis Lucri-Faber Integration
    # =========================================================================

    def get_civis_compatible_output(self) -> Dict[str, Any]:
        """
        Get output in Civis Lucri-Faber compatible format.

        Returns dict with:
          - 'emotion_tensor': [7] tensor for advanced_emotion_integration
          - 'dominant_emotion': str
          - 'arousal': float (derived from velocity + confidence)
          - 'valence': float (derived from emotion type)
          - 'criticality': float
          - 'rppg_heart_rate': optional
          - 'gaze_direction': optional
        """
        if self._latest_result is None:
            return self._default_civis_output()

        result = self._latest_result
        state = self._latest_state

        # Valence mapping (positive vs negative emotions)
        valence_map = {
            'happy': 0.8, 'surprise': 0.3, 'neutral': 0.0,
            'sad': -0.5, 'fear': -0.6, 'angry': -0.7, 'disgust': -0.8,
        }
        valence = valence_map.get(result.dominant_emotion, 0.0)

        # Adjust valence by confidence
        valence *= result.confidence

        # Arousal (from velocity + criticality + emotion intensity)
        arousal = 0.0
        if state is not None:
            arousal = state.emotion_velocity * 5 + state.criticality * 0.5

        # High arousal emotions
        high_arousal_emotions = ['angry', 'fear', 'surprise', 'happy']
        if result.dominant_emotion in high_arousal_emotions:
            arousal += 0.3 * result.confidence

        arousal = np.clip(arousal, 0.0, 1.0)

        return {
            'emotion_tensor': result.emotion_tensor,
            'dominant_emotion': result.dominant_emotion,
            'confidence': result.confidence,
            'valence': valence,
            'arousal': arousal,
            'criticality': state.criticality if state else 0.0,
            'velocity': state.emotion_velocity if state else 0.0,
            'rppg_heart_rate': result.rppg_heart_rate,
            'gaze_direction': result.gaze_direction,
            'blink_rate': result.blink_rate,
            'raw_features': result.raw_features,
        }

    def _default_civis_output(self) -> Dict[str, Any]:
        """Default output when no detection available."""
        return {
            'emotion_tensor': torch.zeros(7),
            'dominant_emotion': 'neutral',
            'confidence': 0.5,
            'valence': 0.0,
            'arousal': 0.0,
            'criticality': 0.0,
            'velocity': 0.0,
            'rppg_heart_rate': None,
            'gaze_direction': None,
            'blink_rate': None,
            'raw_features': None,
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0

        return {
            'frames_processed': self._frame_count,
            'elapsed_time': elapsed,
            'fps': fps,
            'backend': self.backend_name,
        }

    def reset(self):
        """Reset smoothing and statistics."""
        self.smoother.reset()
        self._latest_result = None
        self._latest_state = None
        self._frame_count = 0
        self._start_time = time.time()


# =============================================================================
# Factory Functions
# =============================================================================

def create_cv_bridge(
    backend: str = 'auto',
    censor_model: nn.Module = None,
    device: str = 'cpu',
    **kwargs
) -> CVEmotionBridge:
    """Factory function to create CV emotion bridge."""
    return CVEmotionBridge(
        backend=backend,
        censor_model=censor_model,
        device=device,
        **kwargs
    )


def create_bridge_for_civis(device: str = 'cpu') -> CVEmotionBridge:
    """
    Create bridge optimized for Civis Lucri-Faber integration.

    Uses Censor backend for best integration with bio-inspired agent.
    """
    return CVEmotionBridge(
        backend='censor',
        device=device,
        smoothing_window=7,  # Longer window for stable agent input
        enable_multimodal=True,
    )


# =============================================================================
# Convenience: Quick Detection Function
# =============================================================================

def quick_detect(frame: np.ndarray, backend: str = 'deepface') -> Dict[str, Any]:
    """
    Quick single-frame emotion detection.

    Usage:
        result = quick_detect(cv2.imread('face.jpg'))
        print(result['dominant_emotion'])
    """
    bridge = CVEmotionBridge(backend=backend)
    result = bridge.detect(frame, smooth=False)
    return {
        'emotion': result.dominant_emotion,
        'confidence': result.confidence,
        'probabilities': result.emotion_tensor.tolist(),
    }


__all__ = [
    'CVEmotionBridge',
    'EmotionDetectionResult',
    'TemporalEmotionState',
    'FERBackend',
    'DeepFaceBackend',
    'CensorBackend',
    'create_cv_bridge',
    'create_bridge_for_civis',
    'quick_detect',
]