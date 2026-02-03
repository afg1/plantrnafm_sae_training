"""
Pareto analysis of SAE hyperparameter sweep for PlantRNA-FM.
Generates publication-quality figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_and_clean_data(filepath):
    """Load CSV and clean up column formats."""
    df = pd.read_csv(filepath)
    df.columns = list(df.columns[:-1]) + ['dead_frac']

    def parse_dead_frac(x):
        if pd.isna(x):
            return 0
        if isinstance(x, str):
            if '%' in x:
                return float(x.replace('%', '')) / 100
            return float(x) / 100 if float(x) > 1 else float(x)
        return x / 100 if x > 1 else x
    
    df['dead_frac'] = df['dead_frac'].apply(parse_dead_frac)
    df['effective_features'] = df['dict_size'] - df['dead_features']
    df['fve'] = df['frac_variance_explained']
    df['expansion'] = df['dict_size'] // 480  # d_model = 480
    
    return df


def is_pareto_optimal(costs):
    """Find Pareto-optimal points (minimizing all objectives)."""
    is_optimal = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_optimal[i]:
            is_optimal[is_optimal] = np.any(costs[is_optimal] < c, axis=1) | np.all(costs[is_optimal] == c, axis=1)
            is_optimal[i] = True
    return is_optimal


def plot_fve_vs_sparsity(df, ax):
    """Plot FVE vs k, colored by learning rate."""
    lr_colors = {0.0001: '#2ecc71', 0.00001: '#3498db', 0.000001: '#e74c3c'}
    lr_labels = {0.0001: r'$10^{-4}$', 0.00001: r'$10^{-5}$', 0.000001: r'$10^{-6}$'}
    
    for lr in sorted(df['lr'].unique(), reverse=True):
        subset = df[df['lr'] == lr]
        ax.scatter(subset['k'], subset['fve'], 
                   c=lr_colors.get(lr, 'gray'), 
                   label=lr_labels.get(lr, str(lr)),
                   s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Sparsity (k)')
    ax.set_ylabel('Fraction of Variance Explained')
    ax.set_title('A', loc='left', fontweight='bold', fontsize=14)
    ax.legend(title='Learning Rate', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_xscale('log', base=2)
    ax.set_xticks([32, 64, 128, 256, 512])
    ax.set_xticklabels(['32', '64', '128', '256', '512'])
    ax.set_ylim(0.68, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_fve_vs_effective_features(df, ax):
    """Plot FVE vs effective feature count, colored by learning rate."""
    lr_colors = {0.0001: '#2ecc71', 0.00001: '#3498db', 0.000001: '#e74c3c'}
    lr_labels = {0.0001: r'$10^{-4}$', 0.00001: r'$10^{-5}$', 0.000001: r'$10^{-6}$'}
    
    for lr in sorted(df['lr'].unique(), reverse=True):
        subset = df[df['lr'] == lr]
        ax.scatter(subset['effective_features'], subset['fve'],
                   c=lr_colors.get(lr, 'gray'),
                   label=lr_labels.get(lr, str(lr)),
                   s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Effective Features')
    ax.set_ylabel('Fraction of Variance Explained')
    ax.set_title('B', loc='left', fontweight='bold', fontsize=14)
    ax.legend(title='Learning Rate', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_ylim(0.68, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_fve_vs_dead_features(df, ax):
    """Plot FVE vs dead feature %, colored by k."""
    k_colors = {32: '#9b59b6', 64: '#3498db', 128: '#2ecc71', 256: '#f39c12', 512: '#e74c3c'}
    
    for k in sorted(df['k'].unique()):
        subset = df[df['k'] == k]
        ax.scatter(subset['dead_frac'] * 100, subset['fve'],
                   c=k_colors.get(k, 'gray'),
                   label=f'k={k}',
                   s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Dead Features (%)')
    ax.set_ylabel('Fraction of Variance Explained')
    ax.set_title('C', loc='left', fontweight='bold', fontsize=14)
    ax.legend(title='Sparsity (k)', frameon=True, fancybox=False, edgecolor='gray', ncol=2)
    ax.set_ylim(0.68, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_pareto_frontier(df, ax):
    """Plot Pareto frontier for best LR, with expansion factor markers."""
    best_lr = df[df['lr'] == 0.0001].copy()
    
    # Markers by expansion factor
    exp_markers = {16: 'o', 32: 's', 64: '^'}
    exp_labels = {16: '16× (7.7k)', 32: '32× (15k)', 64: '64× (31k)'}
    
    # Find Pareto optimal (minimize k, minimize 1-FVE)
    costs = np.column_stack([best_lr['k'].values, 1 - best_lr['fve'].values])
    pareto_mask = is_pareto_optimal(costs)
    
    # Plot non-Pareto points
    for exp in sorted(best_lr['expansion'].unique()):
        subset = best_lr[best_lr['expansion'] == exp]
        ax.scatter(subset['k'], subset['fve'],
                   marker=exp_markers.get(exp, 'o'),
                   c='#bdc3c7', s=60, alpha=0.6,
                   label=exp_labels.get(exp))
    
    # Highlight Pareto optimal points
    pareto_points = best_lr[pareto_mask]
    ax.scatter(pareto_points['k'], pareto_points['fve'],
               c='#e74c3c', s=150, marker='*', zorder=5,
               edgecolors='darkred', linewidth=0.5,
               label='Pareto optimal')
    
    # Connect Pareto frontier
    pareto_sorted = pareto_points.sort_values('k')
    ax.plot(pareto_sorted['k'], pareto_sorted['fve'], 
            color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Annotate selected config
    selected = best_lr[(best_lr['k'] == 64) & (best_lr['expansion'] == 32)]
    if len(selected) > 0:
        ax.annotate('Selected\nconfig', 
                    xy=(selected['k'].values[0], selected['fve'].values[0]),
                    xytext=(90, 0.94),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2c3e50'))
    
    ax.set_xlabel('Sparsity (k)')
    ax.set_ylabel('Fraction of Variance Explained')
    ax.set_title('D', loc='left', fontweight='bold', fontsize=14)
    ax.legend(title='Dict Size', frameon=True, fancybox=False, edgecolor='gray', loc='lower right')
    ax.set_xscale('log', base=2)
    ax.set_xticks([32, 64, 128, 256, 512])
    ax.set_xticklabels(['32', '64', '128', '256', '512'])
    ax.set_ylim(0.90, 1.005)
    ax.grid(True, alpha=0.3, linestyle='--')


def create_four_panel_figure(df, output_path):
    """Create the main 4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    plot_fve_vs_sparsity(df, axes[0, 0])
    plot_fve_vs_effective_features(df, axes[0, 1])
    plot_fve_vs_dead_features(df, axes[1, 0])
    plot_pareto_frontier(df, axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_single_pareto_figure(df, output_path):
    """Create a standalone Pareto frontier figure (for main text)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    best_lr = df[df['lr'] == 0.0001].copy()
    
    # Color by dead feature fraction
    scatter = ax.scatter(best_lr['k'], best_lr['fve'],
                         c=best_lr['dead_frac'] * 100,
                         cmap='RdYlGn_r',
                         s=80, alpha=0.9,
                         edgecolors='white', linewidth=0.5,
                         vmin=0, vmax=50)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Dead Features (%)')
    
    # Find and highlight Pareto frontier
    costs = np.column_stack([best_lr['k'].values, 1 - best_lr['fve'].values])
    pareto_mask = is_pareto_optimal(costs)
    pareto_points = best_lr[pareto_mask].sort_values('k')
    
    ax.plot(pareto_points['k'], pareto_points['fve'],
            color='black', linestyle='--', alpha=0.7, linewidth=1.5,
            label='Pareto frontier')
    
    # Annotate selected config
    selected = best_lr[(best_lr['k'] == 64) & (best_lr['expansion'] == 32)]
    if len(selected) > 0:
        ax.scatter(selected['k'], selected['fve'], 
                   s=200, marker='*', c='black', zorder=10)
        ax.annotate('Selected', 
                    xy=(selected['k'].values[0], selected['fve'].values[0]),
                    xytext=(100, 0.945),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    ax.set_xlabel('Sparsity (k)')
    ax.set_ylabel('Fraction of Variance Explained')
    ax.set_xscale('log', base=2)
    ax.set_xticks([32, 64, 128, 256, 512])
    ax.set_xticklabels(['32', '64', '128', '256', '512'])
    ax.set_ylim(0.90, 1.005)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_tables(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("BEST CONFIGURATION PER SPARSITY LEVEL (k)")
    print("="*80)
    
    best_per_k = df.loc[df.groupby('k')['fve'].idxmax()][
        ['k', 'lr', 'expansion', 'fve', 'dead_features', 'dead_frac', 'effective_features']
    ].copy()
    best_per_k['dead_pct'] = (best_per_k['dead_frac'] * 100).round(1)
    best_per_k = best_per_k.sort_values('k')
    
    print(f"{'k':>5} {'LR':>10} {'Exp':>5} {'FVE':>8} {'Dead':>8} {'Dead%':>7} {'Eff.Feat':>10}")
    print("-" * 60)
    for _, row in best_per_k.iterrows():
        print(f"{int(row['k']):>5} {row['lr']:>10.0e} {int(row['expansion']):>5}× "
              f"{row['fve']:>8.4f} {int(row['dead_features']):>8} {row['dead_pct']:>6.1f}% "
              f"{int(row['effective_features']):>10}")
    
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATION")
    print("="*80)
    
    # Best config with <10% dead and >90% FVE
    good = df[(df['dead_frac'] < 0.10) & (df['fve'] > 0.90)].sort_values('fve', ascending=False)
    if len(good) > 0:
        best = good.iloc[0]
        print(f"k={int(best['k'])}, expansion={int(best['expansion'])}× ({int(best['dict_size'])} features), lr={best['lr']:.0e}")
        print(f"FVE: {best['fve']:.4f}, Dead: {best['dead_frac']*100:.2f}%, Effective: {int(best['effective_features'])}")


if __name__ == "__main__":
    # Load data
    df = load_and_clean_data('hyperparam_sweep_results.csv')
    
    # Create figures
    create_four_panel_figure(df, 'pareto_plots_publication.png')
    create_single_pareto_figure(df, 'pareto_single.png')
    
    # Print summaries
    print_summary_tables(df)