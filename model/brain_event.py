# =============================================================================
# Censor -- Brain Event-Driven Mechanism
# =============================================================================
# Inspired by brain neural event-driven architecture:
#   1. Event Bus (Thalamus) - central event routing
#   2. Event Listeners (Brain Regions) - modules that react to events
#   3. Event Types (Neurotransmitters) - different event categories
#   4. Event Memory (Hippocampus) - short-term event storage
#   5. Neuromodulation (Midbrain) - global modulation
#   6. Neural Plasticity Cycle - pruning-based silent/burst cycle
#
# Key innovation: Components communicate via events, not direct calls.
# This enables:
#   - Loose coupling between modules
#   - Event-based attention (like saliency)
#   - Episodic memory formation
#   - Global modulation (like dopamine/norepinephrine)
#   - Neural plasticity cycle (silent ↔ burst ↔ fine → silent)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from enum import Enum, auto
from typing import Dict, List, Callable, Optional, Any
import threading
import math


# =============================================================================
# Part 1: Event Types (Neurotransmitter Analogy)
# =============================================================================

class EventType(Enum):
    """Event types analogous to neurotransmitters."""
    # Excitatory (Glutamate-like)
    SENSORY_INPUT = auto()      # New sensory data arrived
    PREDICTION = auto()        # Model made a prediction
    ATTENTION_SHIFT = auto()     # Attention target changed

    # Modulatory (Dopamine-like)
    REWARD = auto()            # Positive feedback
    PUNISHMENT = auto()       # Negative feedback
    NOVELTY = auto()          # Unexpected input detected
    EXPRESSION_BURST = auto()  # Strong expression detected (plasticity trigger)

    # Plasticity events
    GROWTH_FACTOR = auto()     # BDNF-like growth factor released
    SYNAPSE_PRUNE = auto()    # Synapse pruning event
    SYNAPSE_GROW = auto()     # Synapse growth event

    # Inhibitory (GABA-like)
    SUPPRESSION = auto()       # Suppress certain pathways
    SLEEP = auto()             # Enter low-power mode

    # Global signals
    RESET = auto()             # Reset all states
    SAVE_MEMORY = auto()       # Form episodic memory


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Event:
    """
    Event object analogous to neural spike.

    Attributes:
        type: Event type (neurotransmitter analogy)
        data: Event payload
        priority: Event priority
        source: Source module name
        timestamp: Event creation time
    """

    def __init__(self, event_type: EventType, data: Any,
                 priority: EventPriority = EventPriority.NORMAL,
                 source: str = "system"):
        self.type = event_type
        self.data = data
        self.priority = priority
        self.source = source
        self.timestamp = torch.cuda.Event() if torch.cuda.is_available() else None

    def __repr__(self):
        return f"Event(type={self.type.name}, source={self.source}, priority={self.priority.name})"


# =============================================================================
# Part 2: Event Bus (Thalamus Analog)
# =============================================================================

class EventBus(nn.Module):
    """
    Central event dispatcher - Thalamus analogy.

    Routes events to registered listeners based on event type.
    Supports:
    - Priority-based processing
    - Event filtering
    - Broadcasting to multiple listeners
    """

    def __init__(self, max_queue_size: int = 1000):
        super().__init__()
        self.max_queue_size = max_queue_size
        self.listeners: Dict[EventType, List[Callable]] = {}
        self.event_queue = deque(maxlen=max_queue_size)

        # Event statistics
        self.register_buffer('event_count', torch.zeros(len(EventType)))
        self.register_buffer('last_event_time', torch.zeros(len(EventType)))

    def register(self, event_type: EventType, callback: Callable):
        """Register a listener for an event type."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    def unregister(self, event_type: EventType, callback: Callable):
        """Unregister a listener."""
        if event_type in self.listeners:
            self.listeners[event_type].remove(callback)

    def emit(self, event: Event):
        """
        Emit an event to all registered listeners.

        This is the main entry point for event-driven communication.
        """
        # Add to queue
        self.event_queue.append(event)

        # Update statistics
        type_idx = list(EventType).index(event.type)
        self.event_count[type_idx] += 1

        # Dispatch to listeners
        if event.type in self.listeners:
            for callback in self.listeners[event.type]:
                callback(event)

    def emit_async(self, event: Event):
        """Emit event in a separate thread (non-blocking)."""
        # For async, we just queue the event
        # Actual dispatch happens in forward pass
        self.event_queue.append(event)

    def get_stats(self) -> Dict:
        """Get event statistics."""
        return {
            'total_events': self.event_count.sum().item(),
            'by_type': {
                t.name: self.event_count[list(EventType).index(t)].item()
                for t in EventType
            }
        }


# =============================================================================
# Part 3: Event Listener (Brain Region Analog)
# =============================================================================

class EventListener(nn.Module):
    """
    Base class for event-driven modules - Brain Region analogy.

    Each brain region can:
    - Listen to specific event types
    - Process events asynchronously
    - Maintain local state
    """

    def __init__(self, name: str, listen_types: List[EventType]):
        super().__init__()
        self.name = name
        self.listen_types = listen_types
        self.pending_events: List[Event] = []
        self.local_state: Dict = {}

    def on_event(self, event: Event):
        """Handle an event (override in subclass)."""
        self.pending_events.append(event)

    def clear_pending(self):
        """Clear pending events after processing."""
        self.pending_events.clear()

    def get_local_state(self) -> Dict:
        """Get listener's local state."""
        return self.local_state.copy()


# =============================================================================
# Part 4: Event Memory (Hippocampus Analog)
# =============================================================================

class EventMemory(nn.Module):
    """
    Short-term episodic memory - Hippocampus analogy.

    Stores recent events in a ring buffer:
    - Capacity: N events
    - Retrieval: by similarity or time
    - Consolidation: to long-term storage
    """

    def __init__(self, capacity: int = 100, embedding_dim: int = 256):
        super().__init__()
        self.capacity = capacity
        self.embedding_dim = embedding_dim

        # Ring buffer for events
        self.register_buffer('event_embeddings',
                         torch.zeros(capacity, embedding_dim))
        self.register_buffer('event_types',
                         torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('event_timestamps',
                         torch.zeros(capacity))
        self.register_buffer('head', torch.zeros(1, dtype=torch.long))
        self.register_buffer('size', torch.zeros(1, dtype=torch.long))

        # Memory attention (what to focus on)
        self.memory_attention = nn.Linear(embedding_dim, 1)

    def store(self, embedding: torch.Tensor, event_type: EventType):
        """
        Store an event in memory.

        Args:
            embedding: (D,) event embedding
            event_type: EventType
        """
        idx = self.head.item()

        # Store embedding
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        self.event_embeddings[idx] = embedding[0]

        # Store metadata
        self.event_types[idx] = list(EventType).index(event_type)
        self.event_timestamps[idx] = torch.cuda.Event() if torch.cuda.is_available() else idx

        # Advance head
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> Dict:
        """
        Retrieve top-k similar events.

        Args:
            query: (D,) query embedding
            k: number of events to retrieve

        Returns:
            dict with 'embeddings', 'types', 'scores'
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Compute similarity
        embeddings = self.event_embeddings[:self.size]  # (N, D)
        if embeddings.shape[0] == 0:
            return {'embeddings': [], 'types': [], 'scores': []}

        scores = F.cosine_similarity(query, embeddings, dim=1)  # (N,)

        # Top-k
        top_k = min(k, scores.shape[0])
        scores, idx = torch.topk(scores, top_k)

        return {
            'embeddings': embeddings[idx],
            'types': self.event_types[idx],
            'scores': scores,
            'indices': idx
        }

    def get_recent(self, k: int = 5) -> Dict:
        """Get k most recent events."""
        n = min(k, self.size.item())
        start = (self.head.item() - n) % self.capacity

        indices = torch.arange(start, start + n) % self.capacity

        return {
            'embeddings': self.event_embeddings[indices],
            'types': self.event_types[indices],
            'timestamps': self.event_timestamps[indices]
        }


# =============================================================================
# Part 5: Neuromodulation (Midbrain Analog)
# =============================================================================

class Neuromodulation(nn.Module):
    """
    Global modulation system - Midbrain analogy (Dopamine/Norepinephrine).

    Monitors global state and emits modulation events:
    - Attention modulation
    - arousal level
    - Learning rate adjustment
    """

    def __init__(self, state_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim

        # Modulation levels (like neurotransmitter concentration)
        self.register_buffer('dopamine_level', torch.tensor(0.5))  # Reward prediction
        self.register_buffer('norepinephrine', torch.tensor(0.5))  # Attention
        self.register_buffer('serotonin', torch.tensor(0.5))  # Mood/stability
        self.register_buffer('acetylcholine', torch.tensor(0.5))  # Learning rate

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 neurotransmitter levels
        )

    def forward(self, global_state: torch.Tensor) -> Dict[str, float]:
        """
        Compute modulation levels from global state.

        Args:
            global_state: (B, D) global state representation

        Returns:
            dict of modulation levels
        """
        # Encode state
        levels = self.state_encoder(global_state)  # (B, 4)
        levels = torch.sigmoid(levels)  # (0, 1)

        # Update buffers (detach to avoid graph)
        self.dopamine_level = levels[0, 0].detach()
        self.norepinephrine = levels[0, 1].detach()
        self.serotonin = levels[0, 2].detach()
        self.acetylcholine = levels[0, 3].detach()

        return {
            'dopamine': self.dopamine_level.item(),
            'norepinephrine': self.norepinephrine.item(),
            'serotonin': self.serotonin.item(),
            'acetylcholine': self.acetylcholine.item()
        }

    def get_modulation(self) -> Dict[str, float]:
        """Get current modulation levels."""
        return {
            'dopamine': self.dopamine_level.item(),
            'norepinephrine': self.norepinephrine.item(),
            'serotonin': self.serotonin.item(),
            'acetylcholine': self.acetylcholine.item()
        }


# =============================================================================
# Part 6: Brain Event-Driven Network
# =============================================================================

class BrainEventNetwork(nn.Module):
    """
    Complete brain-inspired event-driven network.

    Integrates:
    - Event Bus (Thalamus)
    - Event Memory (Hippocampus)
    - Neuromodulation (Midbrain)
    - Multiple Event Listeners (Brain Regions)
    """

    def __init__(self, config: Dict = None):
        super().__init__()
        config = config or {}

        state_dim = config.get('state_dim', 128)
        memory_capacity = config.get('memory_capacity', 100)
        embedding_dim = config.get('embedding_dim', 256)

        # Core components
        self.event_bus = EventBus(max_queue_size=config.get('max_queue_size', 1000))
        self.event_memory = EventMemory(capacity=memory_capacity,
                                      embedding_dim=embedding_dim)
        self.neuromodulation = Neuromodulation(state_dim=state_dim)

        # Brain regions (event listeners)
        self.regions: Dict[str, EventListener] = {}

        # Register default handlers
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default event handlers."""

        def handle_reward(event):
            self.neuromodulation.dopamine_level = min(1.0,
                self.neuromodulation.dopamine_level + 0.1)

        def handle_punishment(event):
            self.neuromodulation.dopamine_level = max(0.0,
                self.neuromodulation.dopamine_level - 0.1)

        def handle_novelty(event):
            self.neuromodulation.norepinephrine = min(1.0,
                self.neuromodulation.norepinephrine + 0.2)

        def handle_sensory(event):
            # Store in memory
            if isinstance(event.data, torch.Tensor):
                # Project to embedding_dim if needed
                data = event.data
                if data.shape[-1] != self.event_memory.embedding_dim:
                    # Simple resize via interpolation
                    data = F.adaptive_avg_pool1d(data.unsqueeze(1), self.event_memory.embedding_dim).squeeze(1)
                self.event_memory.store(data, event.type)

        # Register handlers
        self.event_bus.register(EventType.REWARD, handle_reward)
        self.event_bus.register(EventType.PUNISHMENT, handle_punishment)
        self.event_bus.register(EventType.NOVELTY, handle_novelty)
        self.event_bus.register(EventType.SENSORY_INPUT, handle_sensory)

    def register_region(self, name: str, listener: EventListener):
        """Register a brain region (event listener)."""
        self.regions[name] = listener
        for event_type in listener.listen_types:
            self.event_bus.register(event_type, listener.on_event)

    def emit(self, event_type: EventType, data: Any,
            priority: EventPriority = EventPriority.NORMAL,
            source: str = "system"):
        """Emit an event."""
        event = Event(event_type, data, priority, source)
        self.event_bus.emit(event)
        return event

    def forward(self, state: torch.Tensor) -> Dict:
        """
        Process current state through event-driven network.

        Args:
            state: (B, D) current state

        Returns:
            dict with network state and modulation
        """
        # Update neuromodulation
        modulation = self.neuromodulation(state)

        # Process pending events
        for region in self.regions.values():
            for event in region.pending_events:
                # Process event (subclasses override on_event)
                pass
            region.clear_pending()

        return {
            'modulation': modulation,
            'event_stats': self.event_bus.get_stats(),
            'memory_size': self.event_memory.size.item()
        }

    def get_status(self) -> Dict:
        """Get network status."""
        return {
            'modulation': self.neuromodulation.get_modulation(),
            'event_stats': self.event_bus.get_stats(),
            'regions': list(self.regions.keys()),
            'memory': {
                'size': self.event_memory.size.item(),
                'capacity': self.event_memory.capacity
            }
        }


# =============================================================================
# Part 6: Neural Plasticity Cycle (Pruning-Based Silent/Burst Cycle)
# =============================================================================
# Inspired by neural plasticity and synaptic pruning:
#   1. SILENT: Fully sparse / hard frozen (most neurons pruned)
#   2. EVENT_BURST: Strong expression detected → quick response
#   3. GROWTH: Growth factors released → unprune 部分
#   4. FINE_ANALYSIS: Full channels active (TV-L1 + all pathways)
#   5. REGRESS: Return to silent (re-prune unused)
#
# This creates an energy-efficient inference cycle.

class NeuralPlasticityState(Enum):
    """States in the neural plasticity cycle."""
    SILENT = auto()      # Fully pruned / frozen
    EVENT_BURST = auto() # Strong expression detected
    GROWTH = auto()      # Growth factor released
    FINE_ANALYSIS = auto()  # Full analysis active
    REGRESSING = auto() # Returning to silent


class ExpressionDetector(nn.Module):
    """
    Detects strong expression (emotion stimulus) to trigger burst.

    Input: fused features from dual-pathway
    Output: expression intensity in (0, 1)
    """

    def __init__(self, input_dim=1024, threshold=0.7):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold

        # Multi-scale expression detector
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: (B, D) fused features
        Returns:
            intensity: (B, 1) expression intensity
        """
        return self.detector(features)


class PlasticityGate(nn.Module):
    """
    Gate controlling sparse ↔ full transition.

    Uses L0 regularization-style pruning:
    - When intensity < threshold: most channels frozen (sparse)
    - When intensity > threshold: channels activate (full)
    """

    def __init__(self, dim, sparse_ratio=0.1, burst_threshold=0.7):
        super().__init__()
        self.dim = dim
        self.sparse_ratio = sparse_ratio  # Top 10% active in silent mode
        self.burst_threshold = burst_threshold

        # Learnable gate parameters
        self.gamma = nn.Parameter(torch.tensor(-0.5))  # Gate sharpness
        self.zeta = nn.Parameter(torch.tensor(1.0))  # Gate offset

        # Statistics
        self.register_buffer('active_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('total_passes', torch.zeros(1, dtype=torch.long))

    def forward(self, features, expression_intensity):
        """
        Apply plasticity gate.

        Args:
            features: (B, D) input features
            expression_intensity: (B, 1) detected intensity

        Returns:
            gated_features: (B, D) with plasticity applied
            gate_info: dict with gate statistics
        """
        B, D = features.shape

        # Compute gate weight using hard concrete distribution
        # This gives a hard gate (0 or 1) with learnable sharpness
        gate_input = self.gamma * (expression_intensity - self.zeta)
        gate_weight = torch.sigmoid(gate_input / (self.zeta + 1e-8))

        # Determine how many channels to activate
        intensity_val = expression_intensity.mean().item()
        if intensity_val > self.burst_threshold:
            # High intensity: activate more channels
            active_channels = D
            mode = "full"
        else:
            # Low intensity: only sparse channels
            active_channels = max(1, int(D * self.sparse_ratio))
            mode = "sparse"

        # Create sparse mask (top-K selection)
        feature_magnitude = features.abs()
        _, topk_indices = torch.topk(feature_magnitude, active_channels, dim=1)

        # Build mask
        mask = torch.zeros_like(features)
        mask.scatter_(1, topk_indices, 1.0)

        # Apply gate weight modulation
        mask = mask * gate_weight

        gated_features = features * mask

        # Update stats
        self.active_count += active_channels
        self.total_passes += 1

        gate_info = {
            'active_channels': active_channels,
            'total_dim': D,
            'mode': mode,
            'sparse_ratio': active_channels / D,
            'gate_weight': gate_weight.mean().item()
        }

        return gated_features, gate_info


class GrowthFactorController(nn.Module):
    """
    Growth factor controller (BDNF-like).

    When transitioning from SILENT → BURST:
    - Release growth factor to unprune neurons
    - Temporary boost for recovering pathways

    The growth factor follows a temporal pattern:
    - burst_start: High factor (neurons waking up)
    - decay: Factor decays over time
    - baseline: Return to low baseline
    """

    def __init__(self, dim, max_boost=2.0, decay_steps=50):
        super().__init__()
        self.dim = dim
        self.max_boost = max_boost
        self.decay_steps = decay_steps

        # Growth factor state
        self.register_buffer('current_boost', torch.tensor(1.0))
        self.register_buffer('burst_counter', torch.zeros(1, dtype=torch.long))
        self.register_buffer('total_bursts', torch.zeros(1, dtype=torch.long))

    def trigger_burst(self):
        """Trigger a burst event - release growth factor."""
        self.current_boost = torch.tensor(self.max_boost)
        self.burst_counter += 1
        self.total_bursts += 1

    def forward(self, features):
        """
        Apply growth factor to features.

        Args:
            features: (B, D) input features
        Returns:
            boosted_features: (B, D) with growth factor applied
        """
        if self.current_boost > 1.0:
            # Apply boost
            boosted = features * self.current_boost

            # Decay the boost
            self.current_boost = max(1.0,
                self.current_boost - (self.max_boost - 1.0) / self.decay_steps)
        else:
            boosted = features

        return boosted

    def get_boost(self):
        """Get current boost level."""
        return self.current_boost.item()


class TVL1FineAnalyzer(nn.Module):
    """
    TV-L1 fine analysis stage.

    Activated only during FINE_ANALYSIS state.
    Provides edge-preserving smoothing for detailed expression analysis.
    """

    def __init__(self, input_dim=1024):
        super().__init__()
        self.input_dim = input_dim

        # TV-L1 approximation parameters
        self.tv_weight = nn.Parameter(torch.tensor(0.1))
        self.l1_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, features):
        """
        Apply TV-L1 fine analysis.

        In practice, this is approximated via:
        - Total Variation: encourages smooth regions
        - L1: preserves edges

        Args:
            features: (B, D) input features
        Returns:
            refined_features: (B, D) with fine analysis applied
        """
        # Simplified TV-L1: feature refinement via soft thresholding
        tv_w = self.tv_weight.sigmoid()
        l1_w = self.l1_weight.sigmoid()

        # Apply soft shrinkage (L1-like denoising)
        refined = torch.sign(features) * F.relu(features.abs() - l1_w * 0.1)

        return refined


class NeuralPlasticityCycle(nn.Module):
    """
    Complete Neural Plasticity Cycle.

    State Machine:
        SILENT → (expression > threshold) → EVENT_BURST
        EVENT_BURST → (growth factor released) → GROWTH
        GROWTH → (full channels active) → FINE_ANALYSIS
        FINE_ANALYSIS → (no expression) → REGRESSING
        REGRESSING → SILENT

    Key features:
    1. Energy-efficient: Mostly silent with sparse activation
    2. Event-driven: Only full analysis on strong emotions
    3. Plasticity: Channels can be pruned/unpruned dynamically
    4. Fine analysis: TV-L1 edge preservation when needed
    """

    def __init__(self, config: Dict = None):
        super().__init__()
        config = config or {}

        dim = config.get('dim', 1024)
        self.burst_threshold = config.get('burst_threshold', 0.7)
        self.regress_threshold = config.get('regress_threshold', 0.3)
        self.silent_timeout = config.get('silent_timeout', 10)

        # State tracking
        self.state = NeuralPlasticityState.SILENT
        self.register_buffer('state_counter', torch.zeros(1, dtype=torch.long))
        self.register_buffer('last_burst_time', torch.zeros(1, dtype=torch.long))

        # Components
        self.detector = ExpressionDetector(dim, threshold=self.burst_threshold)
        self.gate = PlasticityGate(dim, sparse_ratio=0.1,
                                 burst_threshold=self.burst_threshold)
        self.growth = GrowthFactorController(dim, max_boost=2.0, decay_steps=50)
        self.fine_analyzer = TVL1FineAnalyzer(dim)

        # Statistics
        self.register_buffer('total_cycles', torch.zeros(1))
        self.register_buffer('total_bursts', torch.zeros(1))

    def forward(self, features, enable_fine=False):
        """
        Process through plasticity cycle.

        Args:
            features: (B, D) input features
            enable_fine: whether to enable TV-L1 fine analysis
        Returns:
            output_features: (B, D) processed features
            cycle_info: dict with cycle state info
        """
        # === Step 1: Detect expression intensity ===
        intensity = self.detector(features)  # (B, 1)
        intensity_mean = intensity.mean().item()  # scalar for comparison

        # === Step 2: State machine ===
        if self.state == NeuralPlasticityState.SILENT:
            if intensity_mean > self.burst_threshold:
                self.state = NeuralPlasticityState.EVENT_BURST
                self.growth.trigger_burst()
                self.total_bursts += 1

        elif self.state == NeuralPlasticityState.EVENT_BURST:
            self.state = NeuralPlasticityState.GROWTH

        elif self.state == NeuralPlasticityState.GROWTH:
            if intensity_mean > self.burst_threshold:
                self.state = NeuralPlasticityState.FINE_ANALYSIS

        elif self.state == NeuralPlasticityState.FINE_ANALYSIS:
            if intensity_mean < self.regress_threshold:
                self.state = NeuralPlasticityState.REGRESSING

        elif self.state == NeuralPlasticityState.REGRESSING:
            self.state = NeuralPlasticityState.SILENT
            self.total_cycles += 1

        # Update counter
        self.state_counter += 1

        # === Step 3: Apply components based on state ===
        if self.state == NeuralPlasticityState.SILENT:
            # Most channels frozen
            output, gate_info = self.gate(features, intensity)
            output = output * 0  # Suppress output in silent mode

        elif self.state == NeuralPlasticityState.EVENT_BURST:
            # Quick response: apply growth factor
            output = self.growth(features)
            output, gate_info = self.gate(features, intensity)
            gate_info['mode'] = 'burst'

        elif self.state == NeuralPlasticityState.GROWTH:
            # Growth factor active
            output = self.growth(features)
            output, gate_info = self.gate(features, intensity)

        elif self.state == NeuralPlasticityState.FINE_ANALYSIS:
            # Full analysis: apply TV-L1
            output = self.growth(features)
            output, gate_info = self.gate(features, intensity)

            if enable_fine:
                output = self.fine_analyzer(output)

        else:  # REGRESSING
            output, gate_info = self.gate(features, intensity)
            output = output * 0.5  # Suppress gradually

        cycle_info = {
            'state': self.state.name,
            'intensity': intensity.mean().item(),
            'gate_info': gate_info,
            'growth_boost': self.growth.get_boost(),
            'total_cycles': self.total_cycles.item(),
            'total_bursts': self.total_bursts.item()
        }

        return output, cycle_info

    def reset(self):
        """Reset to SILENT state."""
        self.state = NeuralPlasticityState.SILENT


# =============================================================================
# Part 7: Integration with Censor
# =============================================================================

class EventDrivenCensor(nn.Module):
    """
    Censor with event-driven brain mechanism integration.

    Wraps existing Censor with:
    - Event emission on predictions
    - Feedback integration
    - Memory formation
    """

    def __init__(self, base_model, config: Dict = None):
        super().__init__()
        self.base_model = base_model
        self.config = config or {}

        # Event-driven brain
        self.brain = BrainEventNetwork(config=config)

        # Feedback tracking
        self.register_buffer('correct_count', torch.zeros(1))
        self.register_buffer('total_count', torch.zeros(1))

    def forward(self, x, return_events: bool = False):
        """
        Forward pass with event integration.

        Args:
            x: input features
            return_events: whether to return emitted events

        Returns:
            output: model output
            info: dict with event info
        """
        # Base model forward
        output = self.base_model(x)

        # Emit prediction event
        self.brain.emit(EventType.PREDICTION, x, source="censor")

        # Emit novelty if unexpected (simple heuristic)
        if x.std() > 0.5:
            self.brain.emit(EventType.NOVELTY, x, source="censor")

        # Get brain status
        if x.dim() > 1:
            state = x.mean(dim=0, keepdim=True)
        else:
            state = x.unsqueeze(0)
        brain_state = self.brain(state)

        info = {
            'brain_state': brain_state,
            'modulation': brain_state['modulation']
        }

        if return_events:
            return output, info, self.brain.event_bus.event_queue
        return output, info

    def apply_feedback(self, is_correct: bool):
        """
        Apply user feedback on prediction.

        Args:
            is_correct: whether prediction was correct
        """
        if is_correct:
            self.brain.emit(EventType.REWARD, None, source="feedback")
            self.correct_count += 1
        else:
            self.brain.emit(EventType.PUNISHMENT, None, source="feedback")

        self.total_count += 1

    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if self.total_count.item() == 0:
            return 0.0
        return self.correct_count.item() / self.total_count.item()

    def get_status(self) -> Dict:
        """Get full status."""
        return {
            'accuracy': self.get_accuracy(),
            'brain': self.brain.get_status()
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_brain_event_network(config: Dict = None) -> BrainEventNetwork:
    """Factory function to create event-driven brain network."""
    return BrainEventNetwork(config=config)


def create_event_driven_censor(base_model, config: Dict = None) -> EventDrivenCensor:
    """Factory function to create event-driven Censor."""
    return EventDrivenCensor(base_model, config=config)


def create_neural_plasticity_cycle(config: Dict = None) -> NeuralPlasticityCycle:
    """Factory function to create neural plasticity cycle."""
    return NeuralPlasticityCycle(config=config)