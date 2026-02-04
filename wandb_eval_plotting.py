import polars as pl
import matplotlib.pyplot as plt

# 1. Load Data
df = pl.read_csv("full_model_metrics.csv")

# Ensure sorted by layer
df = df.sort("layer")

# 2. Setup the "Dual Axis" Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Left Axis: Variance (Line Plot) ---
color_var = 'tab:blue'
ax1.set_xlabel('Layer', fontweight='bold')
ax1.set_ylabel('Fraction Variance Explained', color=color_var, fontweight='bold')
ax1.plot(
    df["layer"], 
    df["frac_variance_explained"], 
    color=color_var, 
    marker='o', 
    linewidth=2, 
    label='Variance Exp.'
)
ax1.tick_params(axis='y', labelcolor=color_var)
ax1.set_xticks(df["layer"]) # Force integer ticks for layers
ax1.grid(True, alpha=0.3)

# --- Right Axis: Dead Features (Bar Plot) ---
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color_dead = 'tab:red'
ax2.set_ylabel('Dead Features (Count)', color=color_dead, fontweight='bold')

# Use bars for Dead Features to distinguish them from the line
ax2.bar(
    df["layer"], 
    df["dead_features"], 
    color=color_dead, 
    alpha=0.3, 
    width=0.5, 
    label='Dead Features'
)

# Optional: Add a faint dashed line to help track the bar tops
ax2.plot(df["layer"], df["dead_features"], color=color_dead, linestyle='--', alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color_dead)


plt.tight_layout()
plt.savefig("frac_var_vs_dead_feat.png", dpi=300)
plt.show()