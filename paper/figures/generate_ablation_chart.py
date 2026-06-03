#!/usr/bin/env python3
"""Generate ablation study bar chart for PR paper Figure 2."""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiment_results.json
configs = ['Fast-only', 'Slow-only', 'Dual\nno-MoE', 'No-CASANet', 'No-rPPG', 'Full Model']
accuracy = [85.76, 66.87, 85.28, 77.97, 76.98, 87.74]
std = [19.99, 29.69, 17.94, 19.95, 20.22, 17.05]
f1 = [83.92, 59.30, 81.35, 70.41, 69.49, 83.34]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color scheme
colors = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#B279A2']

# Subplot 1: Accuracy
ax1 = axes[0]
bars1 = ax1.bar(configs, accuracy, yerr=std, capsize=4, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.axhline(y=87.74, color='red', linestyle='--', linewidth=1.5, label='Full Model (87.74%)')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_xlabel('Ablation Configuration', fontsize=12)
ax1.set_title('(a) Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 120)
ax1.legend(loc='upper right')

# Add value labels on bars
for bar, acc in zip(bars1, accuracy):
    height = bar.get_height()
    ax1.annotate(f'{acc:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Subplot 2: F1-Score
ax2 = axes[1]
bars2 = ax2.bar(configs, f1, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.axhline(y=83.34, color='red', linestyle='--', linewidth=1.5, label='Full Model (83.34%)')
ax2.set_ylabel('F1-Score (%)', fontsize=12)
ax2.set_xlabel('Ablation Configuration', fontsize=12)
ax2.set_title('(b) F1-Score Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.legend(loc='upper right')

# Add value labels on bars
for bar, f in zip(bars2, f1):
    height = bar.get_height()
    ax2.annotate(f'{f:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('D:/censor/paper/figures/ablation_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/censor/paper/figures/ablation_chart.pdf', bbox_inches='tight')
print("Ablation chart saved to D:/censor/paper/figures/ablation_chart.png and .pdf")
