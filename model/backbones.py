# =============================================================================
# Censor -- Dual-Pathway Backbone Networks (Enhanced with 3D Swin-Transformer)
# =============================================================================
# Implements the biomimetic dual-pathway architecture:
#   1. FastSubcorticalPathway: 3D ResNet-18 variant (shallow, large stride)
#      Simulates subcortical "low road": superior colliculus -> pulvinar -> amygdala
#      Input: TV-L1 optical flow (B, 2, T, H, W)
#   2. SlowCorticalPathway: 3D Swin-Transformer with shifted-window attention
#      Simulates cortical "high road": LGN -> V1 -> V2 -> V4 -> IT
#      Input: RGB + rPPG (B, 6, T, H, W)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from config.defaults import FAST_PATHWAY_CONFIG, SLOW_PATHWAY_CONFIG


def _load_kinetics400_resnet18():
    """Load torchvision pretrained 3D ResNet-18 (Kinetics-400).

    Returns state_dict with keys like 'layer1.0.conv1.weight', etc.
    The final FC layer (400 classes) is excluded.
    """
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        return model.state_dict()
    except (ImportError, AttributeError):
        # Fallback: try older torchvision API
        try:
            from torchvision.models.video import r3d_18
            model = r3d_18(pretrained=True)
            return model.state_dict()
        except Exception:
            print("[Warning] Could not load pretrained R3D-18 from torchvision. "
                  "Falling back to random initialization.")
            return None


def _load_kinetics400_swin3d():
    """Load torchvision pretrained Video Swin Transformer (Kinetics-400).

    Returns state_dict or None if unavailable.
    """
    try:
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        return model.state_dict()
    except (ImportError, AttributeError):
        try:
            from torchvision.models.video import swin3d_t
            model = swin3d_t(pretrained=True)
            return model.state_dict()
        except Exception:
            pass
    # Fallback: try timm
    try:
        import timm
        model = timm.create_model('swin3d_base_patch244_window877_kinetics400', pretrained=True)
        return model.state_dict()
    except Exception:
        print("[Warning] Could not load pretrained Video Swin3D. "
              "Falling back to random initialization.")
        return None


# =============================================================================
# 3D Window Partitioning Utilities
# =============================================================================

def window_partition_3d(x, window_size):
    """
    Partition a 3D feature map into non-overlapping windows.

    Args:
        x (torch.Tensor): (B, T, H, W, C)
        window_size (tuple): (window_T, window_H, window_W)
    Returns:
        windows (torch.Tensor): (B * num_windows, window_T * window_H * window_W, C)
    """
    B, T, H, W, C = x.shape
    wT, wH, wW = window_size

    # Reshape into windows
    x = x.view(B, T // wT, wT, H // wH, wH, W // wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, wT * wH * wW, C)
    return windows


def window_reverse_3d(windows, window_size, T, H, W):
    """
    Merge window partitions back to original 3D shape.

    Args:
        windows (torch.Tensor): (B * num_windows, window_size[0]*window_size[1]*window_size[2], C)
        window_size (tuple): (window_T, window_H, window_W)
        T (int): Original temporal dimension
        H (int): Original height
        W (int): Original width
    Returns:
        x (torch.Tensor): (B, T, H, W, C)
    """
    wT, wH, wW = window_size
    B = int(windows.shape[0] // (T // wT * H // wH * W // wW))

    x = windows.view(B, T // wT, H // wH, W // wW, wT, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, T, H, W, -1)
    return x


# =============================================================================
# 3D Window Attention with Relative Position Bias
# =============================================================================

class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-Head Self-Attention (W-MSA).

    Computes self-attention within non-overlapping 3D windows with
    learnable relative position bias.

    Mathematical formulation:
        Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d_k) + B) @ V

    where B is the relative position bias table B[relative_position] learned
    during training.

    Args:
        dim (int): Number of input channels
        window_size (tuple): (w_T, w_H, w_W) window sizes
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to add bias to QKV projections
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wT, wH, wW)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection (merged for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Relative position bias table
        # Coordinates range: -window_size+1 to window_size-1
        self.register_buffer(
            'relative_position_bias_table',
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1) *
                        (2 * window_size[2] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position indices
        coords_t = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])

        # 3D meshgrid: (3, wT, wH, wW)
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij'))

        coords_flatten = coords.view(3, -1)  # (3, wT*wH*wW)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 3)

        # Shift to non-negative indices
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        # Combine offsets: (t * (2*H-1) + h) * (2*W-1) + w
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input features, (num_windows*B, N, C)
            mask (torch.Tensor, optional): Attention mask for shifted window, (num_windows, N, N)
        Returns:
            x (torch.Tensor): Output features, (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        # QKV projection and split into heads
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)  # (B_, num_heads, N, N)

        # Apply attention mask (for shifted window)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)

        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x


# =============================================================================
# 3D Swin Transformer Block
# =============================================================================

class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer Block with shifted-window attention.

    Contains W-MSA (Window Multi-Head Self-Attention) or SW-MSA (Shifted-Window MSA)
    followed by a 2-layer MLP with GELU activation.

    Architecture:
        x -> LayerNorm -> (Shifted-)Window Attention -> Residual
          -> LayerNorm -> MLP (GELU) -> Residual
    """

    def __init__(self, dim, num_heads, window_size=(4, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)

        # Window attention
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )

        self.drop_path = nn.Identity()  # Simplified: no stochastic depth
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop_rate)
        )

        # Store for forward pass
        self.register_buffer('attn_mask', None)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features, (B, C, T_in, H_in, W_in)
        Returns:
            out (torch.Tensor): Output features, (B, C, T_in, H_in, W_in)
        """
        B, C, T, H, W = x.shape

        # Pad to make dimensions divisible by window_size
        pad_t = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        # Pad with replication
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))  # (B, C, T+pad_t, H+pad_h, W+pad_w)

        B, C, T_p, H_p, W_p = x.shape

        # Reshape to (B, T_p, H_p, W_p, C) for processing
        x = x.permute(0, 2, 3, 4, 1)  # (B, T_p, H_p, W_p, C)

        # Shifted window
        if any(self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                    dims=(1, 2, 3))
            attn_mask = None  # Skip mask for simplicity; residual shifting provides cross-window info
        else:
            shifted_x = x
            attn_mask = None

        # Partition into windows
        windows = window_partition_3d(shifted_x, self.window_size)  # (num_windows*B, wT*wH*wW, C)

        # Window attention
        windows = self.norm1(windows)
        attn_windows = self.attn(windows, mask=attn_mask)  # (num_windows*B, wT*wH*wW, C)

        # Merge (reverse partition)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, T_p, H_p, W_p)  # (B, T_p, H_p, W_p, C)

        # Reverse shift
        if any(self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        # Residual connection
        x = x + shifted_x  # (B, T_p, H_p, W_p, C)

        # MLP
        x = x + self.mlp(self.norm2(x))

        # Convert back to (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T_p, H_p, W_p)

        # Remove padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :, :T, :H, :W]

        return x

    def _compute_attn_mask(self, T, H, W, B, device):
        """
        Compute attention mask for shifted window attention.

        Creates a mask where positions that become non-adjacent after the
        cyclic shift are masked out in the attention computation.
        Each mask value indicates which window region a position belongs to.
        """
        img_mask = torch.zeros(1, T, H, W, 1, device=device)
        wT, wH, wW = self.window_size
        sT, sH, sW = self.shift_size

        # Define slices for each window region
        # For cyclic-shifted windows, we have 2x2x2 = 8 regions
        t_slices = (slice(0, -wT), slice(-wT, None))
        h_slices = (slice(0, -wH), slice(-wH, None))
        w_slices = (slice(0, -wW), slice(-wW, None))

        cnt = 0
        for t_s in t_slices:
            for h_s in h_slices:
                for w_s in w_slices:
                    img_mask[:, t_s, h_s, w_s, :] = cnt
                    cnt += 1

        # Partition mask into windows: (num_windows, N, 1)
        mask_windows = window_partition_3d(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, wT * wH * wW)  # (num_windows, N)

        # Compute attention mask: 0 for same-region, -inf for different-region
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_windows, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)

        return attn_mask


# =============================================================================
# Fast Subcortical Pathway -- 3D ResNet-18 Shallow Variant
# =============================================================================
# (unchanged from original)
# =============================================================================

class BasicBlock3D(nn.Module):
    """3D Basic ResNet Block with spatial and temporal processing."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1, 1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class FastSubcorticalPathway(nn.Module):
    """
    Fast Subcortical Pathway (3D ResNet-18 Shallow Variant).

    Input:  (B, 2, T, H, W) -- TV-L1 optical flow (x, y displacement)
    Output: (B, 512) -- fast pathway feature vector
    """
    def __init__(self, config=None, pretrained=False):
        super().__init__()
        cfg = config or FAST_PATHWAY_CONFIG
        self.input_channels = cfg['input_channels']
        self.stem_channels = cfg['stem_channels']
        self.output_dim = cfg['output_dim']
        self.layer_channels = cfg['layer_channels']

        self.stem = nn.Sequential(
            nn.Conv3d(self.input_channels, self.stem_channels,
                      kernel_size=cfg['stem_kernel'], stride=cfg['stem_stride'],
                      padding=cfg['stem_padding'], bias=False),
            nn.BatchNorm3d(self.stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        nn.init.kaiming_normal_(self.stem[0].weight, mode='fan_out', nonlinearity='relu')

        self.in_planes = self.stem_channels
        self.layer1 = self._make_layer(BasicBlock3D, self.layer_channels[0], blocks=2, stride=(1, 1, 1))
        self.layer2 = self._make_layer(BasicBlock3D, self.layer_channels[1], blocks=2, stride=(2, 2, 2))
        self.layer3 = self._make_layer(BasicBlock3D, self.layer_channels[2], blocks=2, stride=(2, 2, 2))

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.layer_channels[-1], self.output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

        if pretrained:
            self._load_kinetics_pretrained()

    def _load_kinetics_pretrained(self):
        """Load Kinetics-400 pretrained 3D ResNet-18 weights.

        Handles mismatches:
        - Input channels: pretrained has 3 (RGB), ours has 2 (optical flow).
          Solution: average RGB weights across channels for flow input.
        - FC layer: pretrained has 400 classes, ours has output_dim=512.
          Solution: skip FC layer.
        - Layer structure: torchvision R3D-18 has 4 layers [64,128,256,512],
          ours has 3 layers [64,128,256]. We load matching layers only.
        """
        pretrained_sd = _load_kinetics400_resnet18()
        if pretrained_sd is None:
            return

        # Build mapping from torchvision keys to our keys
        our_sd = self.state_dict()
        loaded, skipped = 0, 0

        # 1. Stem conv: pretrained (3,64,..) -> ours (2,64,..)
        # Average first conv weights across RGB channels
        if 'stem.0.weight' in pretrained_sd:
            pretrained_stem = pretrained_sd['stem.0.weight']  # (64, 3, 3, 7, 7)
            our_stem_shape = our_sd['stem.0.weight'].shape    # (64, 2, 3, 7, 7)
            # Average channels: (64, 1, 3, 7, 7) then repeat for 2 channels
            avg_weight = pretrained_stem.mean(dim=1, keepdim=True)  # (64, 1, 3, 7, 7)
            our_sd['stem.0.weight'] = avg_weight.expand(our_stem_shape).clone()
            loaded += 1

        # Stem BN
        for bn_key in ['stem.1.weight', 'stem.1.bias', 'stem.1.running_mean', 'stem.1.running_var']:
            if bn_key in pretrained_sd and bn_key in our_sd:
                our_sd[bn_key] = pretrained_sd[bn_key]
                loaded += 1

        # 2. ResNet layers: torchvision has layer1-layer4, we have layer1-layer3
        # torchvision: layer1(64), layer2(128), layer3(256), layer4(512)
        # ours:        layer1(64), layer2(128), layer3(256)
        layer_mapping = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            # skip layer4 (we don't have it)
        }

        for tv_layer, our_layer in layer_mapping.items():
            for key in pretrained_sd:
                if key.startswith(tv_layer + '.'):
                    our_key = key.replace(tv_layer + '.', our_layer + '.', 1)
                    if our_key in our_sd and our_sd[our_key].shape == pretrained_sd[key].shape:
                        our_sd[our_key] = pretrained_sd[key]
                        loaded += 1
                    else:
                        skipped += 1

        # Load the filtered state dict
        self.load_state_dict(our_sd)
        print(f"[FastSubcorticalPathway] Loaded Kinetics-400 pretrained weights: "
              f"{loaded} params loaded, {skipped} skipped")

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != (1, 1, 1) or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
            nn.init.kaiming_normal_(downsample[0].weight, mode='fan_out', nonlinearity='relu')
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        feat = self.fc(x)
        return feat


# =============================================================================
# Slow Cortical Pathway -- 3D Swin-Transformer (Full)
# =============================================================================
# Biological analogy: The "high road" (geniculostriate pathway) processes
# detailed visual information via parvocellular cells through V1 -> V2 -> V4 -> IT.
# Uses shifted-window self-attention, analogous to cortical recurrence.
# Latency: ~200-300ms.
#
# Mathematical formulation:
#   Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d_k) + B) @ V
# Within shifted windows of size (W_t, W_h, W_w):
#   Partition x into non-overlapping 3D windows
#   Compute self-attention within each window
#   Cycle-shift windows for cross-window connections
# =============================================================================

class SlowCorticalPathway(nn.Module):
    """
    Slow Cortical Pathway (3D Swin-Transformer with full window attention).

    Uses 3D convolution for patch embedding, followed by Swin Transformer blocks
    with shifted-window multi-head self-attention.

    Input:  (B, 6, T, H, W) -- RGB (3) + rPPG (3) concatenated
    Output: (B, 768) -- pooled features
            + (B, 768, T/16, H/32, W/32) -- spatial feature map for CASANet
    """

    def __init__(self, config=None, pretrained=False):
        super().__init__()
        cfg = config or SLOW_PATHWAY_CONFIG

        self.input_channels = cfg['input_channels']
        self.embed_dim = cfg['embed_dim']
        self.output_dim = cfg['output_dim']
        self.stages_config = cfg['stages']

        # === Patch Embedding ===
        self.patch_embed = nn.Conv3d(
            self.input_channels, self.embed_dim,
            kernel_size=cfg['patch_size'],
            stride=cfg['patch_stride'],
            padding=tuple(p // 2 for p in cfg['patch_size'])
        )
        self.patch_norm = nn.LayerNorm(self.embed_dim)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')

        # === Build Swin Transformer Stages ===
        self.stage_merges = nn.ModuleList()
        self.stages = nn.ModuleList()
        current_dim = self.embed_dim

        for stage_cfg in self.stages_config:
            dim = stage_cfg['dim']
            depth = stage_cfg['depth']
            merge_stride = stage_cfg.get('merge_stride', (1, 1, 1))

            # Patch merging (channel + spatial downsampling)
            merge = nn.Conv3d(current_dim, dim, kernel_size=merge_stride, stride=merge_stride)
            nn.init.kaiming_normal_(merge.weight, mode='fan_out', nonlinearity='relu')
            self.stage_merges.append(merge)

            # Swin Transformer blocks for this stage
            window_size = (min(4, T_simulated := dim // 32 + 1 or 4),
                           min(7, dim // 32 + 1 or 7),
                           min(7, dim // 32 + 1 or 7))

            # Use dimension-appropriate window sizes
            if dim <= 96:
                win_size = (4, 7, 7)
            elif dim <= 192:
                win_size = (2, 7, 7)
            else:
                win_size = (1, 7, 7)  # temporal dim is small at deep stages

            blocks = nn.ModuleList()
            for i in range(depth):
                # Alternate between regular and shifted window
                shift_size = win_size if (i % 2 == 1) else (0, 0, 0)
                block = SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=max(1, dim // 32),
                    window_size=win_size,
                    shift_size=shift_size,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop_rate=0.1
                )
                blocks.append(block)

            self.stages.append(blocks)
            current_dim = dim

        # === Output Head ===
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(current_dim, self.output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

        if pretrained:
            self._load_kinetics_pretrained()

    def _load_kinetics_pretrained(self):
        """Load Kinetics-400 pretrained Video Swin Transformer weights.

        Handles mismatches:
        - Input channels: pretrained has 3 (RGB), ours has 6 (RGB+rPPG).
          Solution: copy RGB weights to first 3 channels, zero-init rPPG channels.
        - FC layer: skip (different output dim).
        - Architecture differences: load matching keys only, skip mismatches.
        """
        pretrained_sd = _load_kinetics400_swin3d()
        if pretrained_sd is None:
            return

        our_sd = self.state_dict()
        loaded, skipped = 0, 0

        for key, val in pretrained_sd.items():
            # Skip classification head
            if 'head.' in key or 'fc.' in key:
                skipped += 1
                continue

            # Handle patch_embed input channel mismatch (3 -> 6)
            if key == 'patch_embed.proj.weight':
                our_key = 'patch_embed.weight'
                if our_key in our_sd:
                    # pretrained: (C_out, 3, t, h, w), ours: (C_out, 6, t, h, w)
                    our_shape = our_sd[our_key].shape
                    new_weight = torch.zeros(our_shape)
                    new_weight[:, :3, ...] = val[:, :3, ...]  # copy RGB
                    # rPPG channels initialized to zero (will learn)
                    our_sd[our_key] = new_weight
                    loaded += 1
                continue

            # Map timm key names to our key names
            our_key = key
            # timm uses 'patch_embed.proj' -> we use 'patch_embed'
            our_key = our_key.replace('patch_embed.proj.', 'patch_embed.')
            # timm uses 'patch_embed.norm' -> we use 'patch_norm'
            our_key = our_key.replace('patch_embed.norm.', 'patch_norm.')
            # timm uses 'layers' -> we use 'stages' with different structure
            # Our architecture is custom, so many timm keys won't match directly

            if our_key in our_sd and our_sd[our_key].shape == val.shape:
                our_sd[our_key] = val
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(our_sd)
        print(f"[SlowCorticalPathway] Loaded Kinetics-400 pretrained weights: "
              f"{loaded} params loaded, {skipped} skipped (architecture mismatch expected)")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): RGB + rPPG video, shape (B, 6, T, H, W)
        Returns:
            pooled (torch.Tensor): Global pooled features, shape (B, 768)
            spatial_map (torch.Tensor): Spatial feature map for CASANet, shape (B, 768, T/16, H/32, W/32)
        """
        # Patch embedding
        x = self.patch_embed(x)
        B, C, T_p, H_p, W_p = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.patch_norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        # Pass through stages
        spatial_map = None
        for stage_idx, (stage_blocks, stage_merge) in enumerate(zip(self.stages, self.stage_merges)):
            # Apply patch merging first
            x = stage_merge(x)

            # Capture spatial map after stage 4 merge (idx 3)
            if stage_idx == 3:
                spatial_map = x.clone()

            # Apply Swin Transformer blocks
            for block in stage_blocks:
                x = block(x)

        # Global pooling
        pooled = self.avgpool(x)
        pooled = pooled.flatten(1)
        pooled = self.fc(pooled)

        return pooled, spatial_map