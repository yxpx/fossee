"""Validate results for data quality"""
import pandas as pd

df = pd.read_csv("data/exports/multi_label_only_detailed.csv")

print("=" * 80)
print("✅ DATA QUALITY VALIDATION")
print("=" * 80)

# Check 1: Missing values
print("\n1️⃣ Missing Values:")
metrics_cols = ['case_a_precision', 'case_b_recall', 'case_c_any_overlap', 'case_d_jaccard']
missing = df[metrics_cols].isna().sum()
print(missing)
if missing.sum() > 0:
    print("   ⚠ WARNING: Found missing values!")
else:
    print("   ✓ No missing values")

# Check 2: Value ranges (should be 0-1)
print("\n2️⃣ Value Range Checks (should be [0, 1]):")
for col in metrics_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    status = "✓" if (min_val >= 0 and max_val <= 1) else "⚠"
    print(f"   {status} {col:20s}: [{min_val:.4f}, {max_val:.4f}]")

# Check 3: Sample counts
print("\n3️⃣ Sample Counts per Model:")
counts = df.groupby('model').size().sort_values(ascending=False)
print(counts)

# Check 4: Edge cases
print("\n4️⃣ Edge Case Statistics:")
print(f"   Empty predictions:        {(df['pred_size'] == 0).sum()}")
print(f"   Empty manual labels:      {(df['manual_size'] == 0).sum()}")
print(f"   Perfect matches (J=1.0):  {(df['case_d_jaccard'] == 1.0).sum()}")
print(f"   No overlap (J=0.0):       {(df['case_d_jaccard'] == 0.0).sum()}")
print(f"   Partial overlap (0<J<1):  {((df['case_d_jaccard'] > 0) & (df['case_d_jaccard'] < 1)).sum()}")

# Check 5: Logical consistency
print("\n5️⃣ Logical Consistency Checks:")
# Case A=1 implies Jaccard > 0
inconsistent_a = df[(df['case_a_precision'] == 1) & (df['case_d_jaccard'] == 0)]
print(f"   Case A=1 but Jaccard=0:   {len(inconsistent_a)} {'✓' if len(inconsistent_a)==0 else '⚠'}")

# Case B=1 implies Jaccard > 0
inconsistent_b = df[(df['case_b_recall'] == 1) & (df['case_d_jaccard'] == 0)]
print(f"   Case B=1 but Jaccard=0:   {len(inconsistent_b)} {'✓' if len(inconsistent_b)==0 else '⚠'}")

# Case C=0 should mean Jaccard=0
inconsistent_c = df[(df['case_c_any_overlap'] == 0) & (df['case_d_jaccard'] > 0)]
print(f"   Case C=0 but Jaccard>0:   {len(inconsistent_c)} {'✓' if len(inconsistent_c)==0 else '⚠'}")

print("\n" + "=" * 80)
print("✅ Validation complete!")
print("=" * 80)
