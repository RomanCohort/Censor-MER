# =============================================================================
# Censor -- Dynamic Action Unit (AU) Decoder
# =============================================================================
# Decodes Action Units (FACS) from fused features with temporal dynamics.
#
# Biological basis: Action Units are fundamental facial muscle movements
# defined by the Facial Action Coding System (FACS). Each AU corresponds to
# specific muscle groups (e.g., AU4 = brow lowerer, AU12 = lip corner puller).
# The temporal dynamics follow an onset-apex-decay pattern.
#
# Mathematical formulation:
#   h_t = BiLSTM( f_fused + PE(t), h_{t-1} )
#   AU_t = sigma( W * h_t + b )
#
# Onset-Peak-Decay (OPD) landmark detection:
#   onset  = min{ t | AU_t > threshold AND AU_t > AU_{t-1} }
#   peak   = argmax_t AU_t  for t in [onset, decay]
#   decay  = max{ t | AU_t > threshold AND AU_t < AU_{t-1} }
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.defaults import AU_DECODER_CONFIG


class DynamicAUDecoder(nn.Module):
    """
    Dynamic AU Decoder with BiLSTM temporal modeling.

    Takes fused pathway features and decodes them into 28 Action Unit
    activation intensities over T temporal steps, with onset-peak-decay
    landmark detection for micro-expression spotting.

    Architecture:
        Input: (B, 1024) fused features
          -> Expand to temporal sequence: (T, B, 1024)
          -> Add temporal positional encoding
          -> BiLSTM(2 layers, 512 hidden, bidirectional)
          -> FC(1024 -> 512) -> ReLU -> Dropout(0.3)
          -> FC(512 -> 28) -> Sigmoid (multi-label)
          -> Output: (B, T, 28) AU intensities + (B, 28, 3) OPD landmarks
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or AU_DECODER_CONFIG

        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.dropout = cfg['dropout']
        self.num_aus = cfg['num_aus']
        self.temporal_steps = cfg['temporal_steps']
        self.threshold = cfg['threshold']

        # Temporal positional encoding
        # Learnable, broadcasts across batch
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, self.temporal_steps, self.input_dim) * 0.02,
            requires_grad=True
        )

        # BiLSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=False  # (T, B, D)
        )

        # Classifier head: 1024 (bidirectional) -> 28 AUs
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_aus)
        )

        # Output activation (Sigmoid for multi-label)
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and FC weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1 (helps gradient flow)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def _detect_opd_landmarks(self, au_sequence):
        """
        Detect Onset-Peak-Decay (OPD) landmarks for each AU across time.

        Args:
            au_sequence (torch.Tensor): AU intensities, shape (B, T, num_aus)
        Returns:
            opd (torch.Tensor): OPD landmarks, shape (B, num_aus, 3)
                - opd[..., 0]: onset frame index
                - opd[..., 1]: peak frame index
                - opd[..., 2]: decay frame index
        """
        B, T, num_aus = au_sequence.shape

        opd = torch.zeros(B, num_aus, 3, device=au_sequence.device, dtype=torch.long)

        for b in range(B):
            for au in range(num_aus):
                intensities = au_sequence[b, :, au]  # (T,)

                # Threshold detection
                above_threshold = intensities > self.threshold  # (T,)

                if above_threshold.sum() == 0:
                    # No activation, set to -1
                    opd[b, au, :] = -1
                    continue

                # Find onset (first frame above threshold with increasing trend)
                onset_idx = 0
                for t in range(T):
                    if above_threshold[t]:
                        if t > 0 and intensities[t] > intensities[t-1]:
                            onset_idx = t
                            break
                        elif t == 0:
                            onset_idx = 0
                            break

                # Find decay (last frame above threshold with decreasing trend)
                decay_idx = T - 1
                for t in range(T - 1, -1, -1):
                    if above_threshold[t]:
                        if t < T - 1 and intensities[t] < intensities[t+1]:
                            decay_idx = t
                            break
                        elif t == T - 1:
                            decay_idx = T - 1
                            break

                # Find peak (maximum intensity between onset and decay)
                peak_idx = torch.argmax(intensities[onset_idx:decay_idx+1]).item() + onset_idx

                opd[b, au, 0] = onset_idx
                opd[b, au, 1] = peak_idx
                opd[b, au, 2] = decay_idx

        return opd

    def forward(self, fused_feat):
        """
        Args:
            fused_feat (torch.Tensor): Fused pathway features, shape (B, 1024)
        Returns:
            au_intensities (torch.Tensor): AU activation intensities, shape (B, T, 28)
            opd (torch.Tensor): Onset-Peak-Decay landmarks, shape (B, 28, 3)
        """
        print(f"[DynamicAUDecoder] Input: {fused_feat.shape}")

        B = fused_feat.shape[0]

        # =====================================================================
        # Expand to temporal sequence
        # =====================================================================
        # (B, 1024) -> (B, T, 1024) by repeating along time with positional encoding
        x = fused_feat.unsqueeze(1).expand(-1, self.temporal_steps, -1)  # (B, T, 1024)
        x = x + self.temporal_pos_encoding  # Add positional encoding

        # Transpose for LSTM: (T, B, D)
        x = x.permute(1, 0, 2)  # (T, B, 1024)
        print(f"[DynamicAUDecoder] Temporal sequence: {x.shape}")

        # =====================================================================
        # BiLSTM forward pass
        # =====================================================================
        lstm_out, _ = self.lstm(x)  # (T, B, 1024) -- 1024 = hidden*2
        print(f"[DynamicAUDecoder] After BiLSTM: {lstm_out.shape}")

        # =====================================================================
        # AU classification
        # =====================================================================
        # Transpose back: (B, T, D)
        lstm_out = lstm_out.permute(1, 0, 2)  # (B, T, 1024)

        au_logits = self.classifier(lstm_out)  # (B, T, 28)
        au_intensities = self.sigmoid(au_logits)  # (B, T, 28) -- multi-label

        print(f"[DynamicAUDecoder] AU intensities: {au_intensities.shape}")

        # =====================================================================
        # OPD landmark detection
        # =====================================================================
        opd = self._detect_opd_landmarks(au_intensities)  # (B, 28, 3)
        print(f"[DynamicAUDecoder] OPD landmarks: {opd.shape}")

        return au_intensities, opd


# =============================================================================
# Alternative: Temporal Transformer Decoder (for future enhancement)
# =============================================================================
# class TemporalTransformerDecoder(nn.Module):
#     """
#     Alternative AU decoder using Temporal Transformer instead of BiLSTM.
#     More expressive but requires more training data.
#     """
#     def __init__(self, input_dim=1024, num_aus=28, num_layers=2, nhead=8, dim_ffn=2048):
#         super().__init__()
#         self.pos_encoding = nn.Parameter(torch.randn(1, 16, input_dim) * 0.02)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim,
#             nhead=nhead,
#             dim_feedforward=dim_ffn,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.classifier = nn.Linear(input_dim, num_aus)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: (B, T, D)
#         x = x + self.pos_encoding
#         x = self.transformer(x)  # (B, T, D)
#         return self.sigmoid(self.classifier(x))  # (B, T, num_aus)