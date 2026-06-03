#!/usr/bin/env python3
"""Generate architecture diagram for PR paper Figure 1."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors for different module types
colors = {
    'input': '#E8F4FD',
    'pathway': '#FFE5B4',
    'fusion': '#E5F5E5',
    'classification': '#F5E5E5',
    'output': '#E5E5F5',
}

def draw_box(ax, x, y, width, height, label, color, fontsize=10):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, start, end, color='black'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=1.5)
    ax.add_patch(arrow)

# Title
ax.text(8, 9.5, 'Censor Architecture Overview (68.35M Parameters)',
        ha='center', fontsize=14, fontweight='bold')

# Input Layer
draw_box(ax, 0.5, 7.5, 2, 1, 'Video Input\n(B×16×224×224)', colors['input'])
draw_box(ax, 3.5, 7.5, 2, 1, 'MTCNN\nFace Detection', colors['input'])
draw_box(ax, 6.5, 7.5, 2, 1, 'TVL1\nOptical Flow', colors['input'])
draw_box(ax, 9.5, 7.5, 2, 1, 'rPPG\nExtractor', colors['input'])

# Arrows from input
draw_arrow(ax, (1.5, 7.5), (1.5, 6.5))
draw_arrow(ax, (4.5, 7.5), (4.5, 6.5))
draw_arrow(ax, (7.5, 7.5), (3, 5.5))  # Optical flow to fast pathway
draw_arrow(ax, (10.5, 7.5), (10.5, 6.5))

# Preprocessing
draw_box(ax, 0.5, 6, 2.5, 1, 'SaliencyDetector\n(0.12M)', colors['pathway'])

# Dual Pathway Layer
draw_box(ax, 1, 4.5, 4, 1.5, 'Fast Pathway\n3D ResNet-18\n(12.85M)', colors['pathway'])
draw_box(ax, 9, 4.5, 4, 1.5, 'Slow Pathway\n3D Swin-T\n(31.40M)', colors['pathway'])

ax.text(1, 5.5, 'Optical Flow (B×2×16×224×224)', fontsize=8, va='bottom')
ax.text(9, 5.5, 'RGB + rPPG (B×6×16×224×224)', fontsize=8, va='bottom')

# Arrows to fusion
draw_arrow(ax, (3, 4.5), (5, 3.5))
draw_arrow(ax, (11, 4.5), (9, 3.5))

# Fusion Layer
draw_box(ax, 5, 2.5, 2, 1, 'AmygdalaGate\n(0.08M)', colors['fusion'])
draw_box(ax, 8, 2.5, 2, 1, 'FFA Fusion\n(1.64M)', colors['fusion'])
draw_box(ax, 6, 1, 3, 1, 'TSFmicroFusion\n(4.38M)', colors['fusion'])

draw_arrow(ax, (6, 2.5), (7.5, 1.5))
draw_arrow(ax, (9, 2.5), (7.5, 1.5))

# Temporal Attention
draw_box(ax, 12, 2.5, 2.5, 1.5, 'CASANet\n(2.12M)', colors['fusion'])
draw_arrow(ax, (7.5, 1.5), (12.5, 2.5))

# AU Decoder (parallel)
draw_box(ax, 13.5, 5, 2, 1, 'AU Decoder\n(8.45M)\n28 AUs', colors['classification'])
draw_arrow(ax, (10, 4.5), (13.5, 5))

# Classification Layer
draw_box(ax, 5, 0, 4, 1, 'MoE Head\n(7.31M)\n3 Experts', colors['classification'])
draw_arrow(ax, (7.5, 1), (7, 1))

draw_box(ax, 10.5, 0, 2.5, 1, 'Sparse Control\n(Proposed)', '#F0F0F0')
draw_arrow(ax, (7, 0.5), (10.5, 0.5), color='gray')

# Output
draw_box(ax, 5, -0.5, 3, 0.5, 'ME Classification\n(4 classes)', colors['output'])
draw_arrow(ax, (7, 0), (6.5, -0.5))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input/Preprocessing'),
    mpatches.Patch(facecolor=colors['pathway'], edgecolor='black', label='Pathway Processing'),
    mpatches.Patch(facecolor=colors['fusion'], edgecolor='black', label='Fusion/Attention'),
    mpatches.Patch(facecolor=colors['classification'], edgecolor='black', label='Classification'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Neuroscience analogy annotation
ax.text(2, 4, 'Subcortical\n(analogy)', fontsize=9, ha='center', style='italic', color='gray')
ax.text(11, 4, 'Cortical\n(analogy)', fontsize=9, ha='center', style='italic', color='gray')

plt.tight_layout()
plt.savefig('D:/censor/paper/figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/censor/paper/figures/architecture_diagram.pdf', bbox_inches='tight')
print("Architecture diagram saved to D:/censor/paper/figures/")