import pandas as pd
import ast
from pathlib import Path
from collections import defaultdict

print("="*70)
print("📊 CONFUSION MATRIX ANALYSIS")
print("="*70)

# Load consolidated data
df = pd.read_csv("data/processed/yaksh_100_qns_consolidated.csv")
print(f"\n✅ Loaded: {df.shape}")

# Parse manual categories
def parse_categories(cat_str):
    if pd.isna(cat_str) or cat_str == '':
        return []
    if isinstance(cat_str, str):
        if cat_str.startswith('['):
            return ast.literal_eval(cat_str)
        else:
            return [c.strip() for c in cat_str.split(',') if c.strip()]
    return []

df['manual_cats_list'] = df['manual_categories'].apply(parse_categories)

# Get all categories
all_cats = set()
for cats in df['manual_cats_list']:
    all_cats.update(cats)
all_cats = sorted([c for c in all_cats if c])

print(f"📋 Categories: {all_cats}")

# Best model (from previous analysis - adjust if needed)
best_model = 'openai_gpt-5.2'
print(f"\n🏆 Analyzing best model: {best_model}")

# Get all columns for this model
model_cols = [c for c in df.columns if c.startswith('results_') and best_model in c]
print(f"   Found {len(model_cols)} prediction columns")

# Build confusion matrix
confusion = defaultdict(lambda: defaultdict(int))

for idx, row in df.iterrows():
    manual = row['manual_cats_list']
    
    if not manual:
        continue
    
    # For multi-label, take first category as primary
    manual_primary = manual[0] if manual else 'UNKNOWN'
    
    # Check all model predictions
    for col in model_cols:
        pred = str(row[col]).strip()
        
        # Record prediction
        confusion[manual_primary][pred] += 1

# Create confusion matrix DataFrame
confusion_rows = []

for manual_cat in all_cats:
    row_data = {'Manual_Label': manual_cat}
    
    for pred_cat in all_cats:
        row_data[f'Predicted_{pred_cat}'] = confusion[manual_cat].get(pred_cat, 0)
    
    # Add "Other" category for predictions not in standard categories
    other_count = sum(v for k, v in confusion[manual_cat].items() if k not in all_cats)
    row_data['Predicted_Other'] = other_count
    
    confusion_rows.append(row_data)

confusion_df = pd.DataFrame(confusion_rows)

# Save
Path("data/exports").mkdir(parents=True, exist_ok=True)
confusion_df.to_csv("data/exports/api_confusion_matrix.csv", index=False)

print("\n" + "="*70)
print(f"📊 CONFUSION MATRIX: {best_model}")
print("="*70)
print("\n" + str(confusion_df))

# Find most common mistakes
print("\n" + "="*70)
print("📊 Most Common Mistakes")
print("="*70)

mistakes = []
for manual_cat in all_cats:
    for pred_cat in all_cats:
        if manual_cat != pred_cat:
            count = confusion[manual_cat].get(pred_cat, 0)
            if count > 0:
                mistakes.append({
                    'Manual': manual_cat,
                    'Predicted_As': pred_cat,
                    'Count': count
                })

mistakes_df = pd.DataFrame(mistakes).sort_values('Count', ascending=False)
print("\nTop 10 confusion pairs:")
print(mistakes_df.head(10).to_string(index=False))

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated: data/exports/api_confusion_matrix.csv")
