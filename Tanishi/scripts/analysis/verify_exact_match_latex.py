import pandas as pd
import numpy as np

# Load the multi-label detailed data
df = pd.read_csv('data\exports\multi_label_only_detailed.csv')

print("="*70)
print("EXACT MATCH ANALYSIS - Verification Script")
print("="*70)

# 1. Calculate Exact Match (Case A = 1.0 AND Case B = 1.0)
df['exact_match'] = ((df['case_a_precision'] == 1.0) & 
                     (df['case_b_recall'] == 1.0)).astype(int)

# 2. Calculate metrics by model and run
summary_by_run = df.groupby(['model', 'run']).agg({
    'case_a_precision': ['mean', 'std'],
    'case_b_recall': ['mean', 'std'],
    'case_c_any_overlap': ['mean', 'std'],
    'case_d_jaccard': ['mean', 'std'],
    'exact_match': ['mean', 'std', 'sum', 'count']
}).round(4)

summary_by_run.columns = ['_'.join(col).strip() for col in summary_by_run.columns.values]
summary_by_run = summary_by_run.reset_index()

# 3. Calculate overall metrics by model (averaged across runs)
summary_by_model = df.groupby('model').agg({
    'case_a_precision': 'mean',
    'case_b_recall': 'mean',
    'case_c_any_overlap': 'mean',
    'case_d_jaccard': 'mean',
    'exact_match': ['mean', 'sum', 'count']
}).round(4)

summary_by_model.columns = ['_'.join(col).strip() if col[1] else col[0] 
                             for col in summary_by_model.columns.values]
summary_by_model = summary_by_model.reset_index()

# 4. Calculate OVERALL averages (for the LaTeX table)
overall_stats = {
    'metric': ['Exact Match', 'Case A (Precision)', 'Case B (Recall)', 
               'Case C (Any Overlap)', 'Case D (Jaccard)'],
    'average_score': [
        df['exact_match'].mean(),
        df['case_a_precision'].mean(),
        df['case_b_recall'].mean(),
        df['case_c_any_overlap'].mean(),
        df['case_d_jaccard'].mean()
    ]
}

comparison_df = pd.DataFrame(overall_stats)
comparison_df['average_score_pct'] = (comparison_df['average_score'] * 100).round(2)

# Calculate difference from exact match baseline
exact_match_baseline = comparison_df.loc[comparison_df['metric'] == 'Exact Match', 'average_score'].values[0]
comparison_df['vs_exact_match_pp'] = ((comparison_df['average_score'] - exact_match_baseline) * 100).round(2)
comparison_df['vs_exact_match_display'] = comparison_df['vs_exact_match_pp'].apply(
    lambda x: 'Baseline' if x == 0 else f'+{x:.1f} pp'
)

# 5. Save to CSVs
summary_by_run.to_csv('exact_match_by_model_run.csv', index=False)
summary_by_model.to_csv('exact_match_by_model_summary.csv', index=False)
comparison_df.to_csv('exact_match_comparison_table.csv', index=False)

# 6. Print results for verification
print("\n" + "="*70)
print("TABLE 15 DATA - For LaTeX Verification")
print("="*70)
print(comparison_df[['metric', 'average_score_pct', 'vs_exact_match_display']].to_string(index=False))

print("\n" + "="*70)
print("EXACT MATCH BY MODEL (Averaged Across Runs)")
print("="*70)
print(summary_by_model[['model', 'exact_match_mean', 'exact_match_sum', 'exact_match_count']].to_string(index=False))

# 7. Detailed breakdown for each model
print("\n" + "="*70)
print("DETAILED BREAKDOWN BY MODEL AND RUN")
print("="*70)
print(summary_by_run[['model', 'run', 'exact_match_mean', 'exact_match_sum', 
                       'case_a_precision_mean', 'case_b_recall_mean', 
                       'case_c_any_overlap_mean', 'case_d_jaccard_mean']].to_string(index=False))

# 8. Verify the LaTeX numbers
print("\n" + "="*70)
print("LATEX TABLE VERIFICATION")
print("="*70)
latex_check = {
    'LaTeX Value': ['8.7%', '25.0%', '29.0%', '83.2%', '0.410 (41.0%)'],
    'Actual Value': [
        f"{df['exact_match'].mean()*100:.2f}%",
        f"{df['case_a_precision'].mean()*100:.2f}%",
        f"{df['case_b_recall'].mean()*100:.2f}%",
        f"{df['case_c_any_overlap'].mean()*100:.2f}%",
        f"{df['case_d_jaccard'].mean():.3f} ({df['case_d_jaccard'].mean()*100:.2f}%)"
    ],
    'Match?': ['❌', '✅', '✅', '✅', '✅']
}

# Update match status
exact_match_actual = df['exact_match'].mean() * 100
case_a_actual = df['case_a_precision'].mean() * 100
case_b_actual = df['case_b_recall'].mean() * 100
case_c_actual = df['case_c_any_overlap'].mean() * 100
case_d_actual = df['case_d_jaccard'].mean()

latex_check['Match?'] = [
    '✅' if abs(8.7 - exact_match_actual) < 0.5 else '❌ WRONG',
    '✅' if abs(25.0 - case_a_actual) < 0.5 else '❌ WRONG',
    '✅' if abs(29.0 - case_b_actual) < 0.5 else '❌ WRONG',
    '✅' if abs(83.2 - case_c_actual) < 0.5 else '❌ WRONG',
    '✅' if abs(0.410 - case_d_actual) < 0.01 else '❌ WRONG'
]

verification_df = pd.DataFrame(latex_check)
print(verification_df.to_string(index=False))

print("\n" + "="*70)
print("VERDICT")
print("="*70)
if latex_check['Match?'][0] == '❌ WRONG':
    print(f"⚠️  EXACT MATCH IS INCORRECT!")
    print(f"   LaTeX says: 8.7%")
    print(f"   Actual data: {exact_match_actual:.2f}%")
    print(f"   Difference: {abs(8.7 - exact_match_actual):.2f} pp")
    print(f"\n   CORRECTED LaTeX should be:")
    print(f"   Exact Match: {exact_match_actual:.1f}%")
    print(f"   Case A: +{(case_a_actual - exact_match_actual):.1f} pp")
    print(f"   Case B: +{(case_b_actual - exact_match_actual):.1f} pp")
    print(f"   Case C: +{(case_c_actual - exact_match_actual):.1f} pp")
    print(f"   Case D: +{(case_d_actual*100 - exact_match_actual):.1f} pp")
else:
    print("✅ All LaTeX values are CORRECT!")

print("\n" + "="*70)
print("FILES GENERATED:")
print("="*70)
print("1. exact_match_by_model_run.csv - Detailed by model and run")
print("2. exact_match_by_model_summary.csv - Summary by model")
print("3. exact_match_comparison_table.csv - Data for Table 15")
print("="*70)
