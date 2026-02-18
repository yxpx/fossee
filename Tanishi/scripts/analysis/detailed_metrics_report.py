"""Generate detailed report with all 4 metrics"""
import pandas as pd

df = pd.read_csv("data/exports/multi_label_only_detailed.csv")

# Summary by model
summary = df.groupby('model').agg({
    'case_a_precision': ['mean', 'std'],
    'case_b_recall': ['mean', 'std'],
    'case_c_any_overlap': ['mean', 'std'],
    'case_d_jaccard': ['mean', 'std', 'count']
}).round(4)

print("=" * 100)
print("DETAILED METRICS - ALL 4 CASES (Multi-Label Only)")
print("=" * 100)
print("\n", summary.to_string())

# Best model per metric
print("\n" + "=" * 100)
print("🏆 BEST PERFORMER PER METRIC:")
print("=" * 100)

metrics = {
    'Case A (Precision)': 'case_a_precision',
    'Case B (Recall)': 'case_b_recall',
    'Case C (Any Overlap)': 'case_c_any_overlap',
    'Case D (Jaccard)': 'case_d_jaccard'
}

for name, col in metrics.items():
    best_model = df.groupby('model')[col].mean().idxmax()
    best_score = df.groupby('model')[col].mean().max()
    print(f"{name:25s}: {best_model:30s} → {best_score:.4f}")

# Export for LaTeX
summary.to_csv("data/exports/multi_label_all_4_metrics.csv")
print("\n✓ Saved: data/exports/multi_label_all_4_metrics.csv")
