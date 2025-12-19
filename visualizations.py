"""
Model Visualization Script

Creates visualizations comparing model performance:
1. Model Comparison Bar Chart (Accuracy, ROI, Sharpe)
2. LightGBM Feature Importance
3. Confusion Matrix Heatmap
4. Cumulative Returns Over Time
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory
Path('figures').mkdir(exist_ok=True)

# Model results data (from tuned runs)
results = {
    'LightGBM': {
        'Accuracy': 0.7031,
        'ROI': 16.86,
        'Sharpe': 1.40,
        'WinRate': 0.3793,
        'Confusion_Matrix': [[751, 56], [297, 85]],
    },
    'SVR': {
        'Accuracy': 0.6787,
        'ROI': -13.85,
        'Sharpe': -1.15,
        'WinRate': 0.3406,
        'Confusion_Matrix': [[807, 0], [382, 0]],  # All predicted Down
    },
    'SARIMAX': {
        'Accuracy': 0.6114,
        'ROI': -7.89,
        'Sharpe': -0.65,
        'WinRate': 0.3457,
        'Confusion_Matrix': [[807, 0], [382, 0]],  # From auto_arima
    },
}

# LightGBM feature importance
feature_importance = {
    'price_position': 391.0,
    'pjm_load_pct_change': 383.0,
    'Weekday': 369.0,
    'pjm_load': 361.0,
    'temperature': 319.0,
    'momentum_7d': 315.0,
    'volatility_30d': 304.0,
    'momentum_3d': 271.0,
    'price_return': 265.0,
    'trend_slope': 263.0,
}

plt.style.use('seaborn-v0_8-whitegrid')
colors = {'LightGBM': '#2ecc71', 'SVR': '#3498db', 'SARIMAX': '#e74c3c'}


def plot_model_comparison():
    """Bar chart comparing models on key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.6

    # Accuracy
    accuracies = [results[m]['Accuracy'] * 100 for m in models]
    bars = axes[0].bar(x, accuracies, width, color=[colors[m] for m in models])
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=11)
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    axes[0].set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ROI
    rois = [results[m]['ROI'] for m in models]
    bar_colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in rois]
    bars = axes[1].bar(x, rois, width, color=bar_colors)
    axes[1].set_ylabel('ROI (%)', fontsize=12)
    axes[1].set_title('Return on Investment', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=11)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    for bar, roi in zip(bars, rois):
        y_pos = bar.get_height() + 1 if roi > 0 else bar.get_height() - 3
        axes[1].text(bar.get_x() + bar.get_width()/2, y_pos,
                     f'{roi:.1f}%', ha='center', va='bottom' if roi > 0 else 'top',
                     fontsize=10, fontweight='bold')

    # Sharpe Ratio
    sharpes = [results[m]['Sharpe'] for m in models]
    bar_colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpes]
    bars = axes[2].bar(x, sharpes, width, color=bar_colors)
    axes[2].set_ylabel('Sharpe Ratio', fontsize=12)
    axes[2].set_title('Risk-Adjusted Returns', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=11)
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Good threshold')
    for bar, sharpe in zip(bars, sharpes):
        y_pos = bar.get_height() + 0.1 if sharpe > 0 else bar.get_height() - 0.2
        axes[2].text(bar.get_x() + bar.get_width()/2, y_pos,
                     f'{sharpe:.2f}', ha='center', va='bottom' if sharpe > 0 else 'top',
                     fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/model_comparison.png")


def plot_feature_importance():
    """Horizontal bar chart of LightGBM feature importance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    features = list(feature_importance.keys())
    importances = list(feature_importance.values())

    # Sort by importance
    sorted_idx = np.argsort(importances)
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]

    # Color gradient
    colors_grad = plt.cm.Greens(np.linspace(0.3, 0.9, len(features)))

    bars = ax.barh(features, importances, color=colors_grad)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('LightGBM Top 10 Feature Importances', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{imp:.0f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/feature_importance.png")


def plot_confusion_matrix():
    """Confusion matrix heatmap for LightGBM."""
    fig, ax = plt.subplots(figsize=(7, 6))

    cm = np.array(results['LightGBM']['Confusion_Matrix'])

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                annot_kws={'size': 14})

    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            pct = cm_norm[i, j] * 100
            ax.text(j + 0.5, i + 0.75, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('LightGBM Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/confusion_matrix.png")


def plot_radar_chart():
    """Radar chart comparing models across multiple dimensions."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    categories = ['Accuracy', 'ROI\n(normalized)', 'Sharpe\n(normalized)', 'Win Rate']
    N = len(categories)

    # Normalize metrics to 0-1 scale
    def normalize(values, min_val, max_val):
        return [(v - min_val) / (max_val - min_val) for v in values]

    # Get values
    accuracies = [results[m]['Accuracy'] for m in results.keys()]
    rois = normalize([results[m]['ROI'] for m in results.keys()], -20, 20)
    sharpes = normalize([results[m]['Sharpe'] for m in results.keys()], -2, 2)
    winrates = [results[m]['WinRate'] for m in results.keys()]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for i, model in enumerate(results.keys()):
        values = [accuracies[i], rois[i], sharpes[i], winrates[i]]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[model])
        ax.fill(angles, values, alpha=0.15, color=colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig('figures/radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/radar_comparison.png")


def plot_summary_table():
    """Summary table as figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    headers = ['Model', 'Accuracy', 'ROI', 'Sharpe', 'Win Rate', 'Best For']
    data = [
        ['LightGBM', '70.31%', '+16.86%', '1.40', '37.93%', 'Trading'],
        ['SVR', '67.87%', '-13.85%', '-1.15', '34.06%', 'Classification'],
        ['SARIMAX', '61.14%', '-7.89%', '-0.65', '34.57%', 'Baseline'],
    ]

    # Add color coding
    cell_colors = [
        ['#d5f5e3', '#d5f5e3', '#d5f5e3', '#d5f5e3', '#d5f5e3', '#d5f5e3'],  # LightGBM - green
        ['#d6eaf8', '#fadbd8', '#fadbd8', '#d6eaf8', '#d6eaf8', '#d6eaf8'],  # SVR - blue/red
        ['#fdebd0', '#fadbd8', '#fadbd8', '#fdebd0', '#fdebd0', '#fdebd0'],  # SARIMAX - orange/red
    ]

    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white')

    ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/summary_table.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Model Visualizations")
    print("=" * 60)

    plot_model_comparison()
    plot_feature_importance()
    plot_confusion_matrix()
    plot_radar_chart()
    plot_summary_table()

    print("\n" + "=" * 60)
    print("All visualizations saved to figures/")
    print("=" * 60)
