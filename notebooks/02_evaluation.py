"""
02_evaluation.py — Compare models side by side
Run: python notebooks/02_evaluation.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, NormalPredictor
from surprise.model_selection import cross_validate

# ── Load Data ──────────────────────────────────────────────────────────────────
print("Loading ratings...")
ratings = pd.read_csv('data/ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'],
                      encoding='latin-1')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# ── Baseline: Random / Average prediction ─────────────────────────────────────
print("\nEvaluating Baseline model...")
baseline = NormalPredictor()
baseline_results = cross_validate(baseline, data, measures=['RMSE', 'MAE'],
                                  cv=3, verbose=False)

baseline_rmse = baseline_results['test_rmse'].mean()
baseline_mae  = baseline_results['test_mae'].mean()

print(f"  Baseline RMSE: {baseline_rmse:.4f}")
print(f"  Baseline MAE : {baseline_mae:.4f}")

# ── SVD Collaborative Filtering ────────────────────────────────────────────────
print("\nEvaluating SVD model (this takes a few minutes)...")
svd = SVD(n_factors=100, n_epochs=20, random_state=42)
svd_results = cross_validate(svd, data, measures=['RMSE', 'MAE'],
                             cv=3, verbose=False)

svd_rmse = svd_results['test_rmse'].mean()
svd_mae  = svd_results['test_mae'].mean()

print(f"  SVD RMSE: {svd_rmse:.4f}")
print(f"  SVD MAE : {svd_mae:.4f}")

# ── Summary Table ──────────────────────────────────────────────────────────────
print("\n===== MODEL COMPARISON =====")
print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8}")
print("-" * 43)
print(f"{'Baseline (random)':<25} {baseline_rmse:>8.4f} {baseline_mae:>8.4f}")
print(f"{'SVD Collaborative':<25} {svd_rmse:>8.4f} {svd_mae:>8.4f}")
improvement = ((baseline_rmse - svd_rmse) / baseline_rmse) * 100
print(f"\nSVD improved RMSE by {improvement:.1f}% over baseline.")

# ── Bar Chart ─────────────────────────────────────────────────────────────────
models = ['Baseline', 'SVD']
rmse_scores = [baseline_rmse, svd_rmse]
mae_scores  = [baseline_mae,  svd_mae]

x = range(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar([i - width/2 for i in x], rmse_scores, width, label='RMSE', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], mae_scores,  width, label='MAE',  color='coral')

ax.set_title('Model Comparison: RMSE and MAE (lower is better)')
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylabel('Error Score')
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('notebooks/plot_model_comparison.png')
plt.show()
print("\nSaved: plot_model_comparison.png")