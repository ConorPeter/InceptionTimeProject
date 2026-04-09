"""
Generate report figures:
  figures/experiment_barchart.png  - accuracy comparison across all experiments
  figures/combined_loss_curve.png  - train vs val loss for the combined run (both datasets)
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ['GunPoint', 'ECG200']

# All accuracy values — paper + all experiments
# Screening experiments ran on improvement/* branches; values from their df_metrics.csv.
ACCURACIES = {
    'GunPoint': {
        'Paper\n(ensemble)':        0.990,
        'Baseline\n(1500ep)':       0.9933,
        'Augmentation\n(500ep)':    0.9800,
        'Dropout\n(500ep)':         0.9467,
        'Early\nStopping\n(500ep)': 0.8867,
        'Combined\n(1500ep)':       0.9933,
    },
    'ECG200': {
        'Paper\n(ensemble)':        0.880,
        'Baseline\n(1500ep)':       0.9200,
        'Augmentation\n(500ep)':    0.9200,
        'Dropout\n(500ep)':         0.8900,
        'Early\nStopping\n(500ep)': 0.7100,
        'Combined\n(1500ep)':       0.9100,
    },
}

# FordA — separate panel with zoomed y-axis (values are tightly clustered)
FORDA_ACCURACIES = {
    'Baseline\n(50 epochs)':      0.9561,
    'Larger Kernel\n(50 epochs)': 0.9568,
}

LABELS = list(ACCURACIES['GunPoint'].keys())
COLORS = ['#888888', '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']


def draw_bars(ax, labels, values, colors, ylim, offset=0.003):
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=8
        )
    ax.set_ylim(*ylim)
    ax.tick_params(axis='x', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Figure 1a: GunPoint & ECG200 bar chart ───────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

for ax, dataset in zip(axes, DATASETS):
    values = [ACCURACIES[dataset][l] for l in LABELS]
    draw_bars(ax, LABELS, values, COLORS, ylim=(0.6, 1.02))
    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy')

fig.suptitle('Test Accuracy by Experiment', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, 'experiment_barchart.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')


# ── Figure 1b: FordA kernel experiment bar chart ─────────────────────────────

fig, ax = plt.subplots(figsize=(4, 4))

forda_labels = list(FORDA_ACCURACIES.keys())
forda_values = list(FORDA_ACCURACIES.values())
forda_colors = ['#4C72B0', '#55A868']

draw_bars(ax, forda_labels, forda_values, forda_colors, ylim=(0.950, 0.960), offset=0.0003)
ax.set_title('FordA — Kernel Size Experiment', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy')

fig.tight_layout()
out = os.path.join(FIGURES_DIR, 'forda_barchart.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')


# ── Figure 2: Combined loss curves (GunPoint + ECG200 side by side) ───────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, dataset in zip(axes, DATASETS):
    history_path = os.path.join(ROOT, 'results_combined', dataset, 'history.csv')
    hist = pd.read_csv(history_path)

    loss_key = 'loss'
    val_loss_key = 'val_loss'

    ax.plot(hist[loss_key].values,     label='Train loss', linewidth=1.2)
    ax.plot(hist[val_loss_key].values, label='Val loss',   linewidth=1.2, linestyle='--')

    best_epoch = hist[val_loss_key].idxmin()
    ax.axvline(best_epoch, color='red', linewidth=0.8, linestyle=':', label=f'Best epoch ({best_epoch})')

    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Training vs Validation Loss — Combined Model (Aug + Dropout, 1500ep)',
             fontsize=11, fontweight='bold', y=1.01)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, 'combined_loss_curve.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
