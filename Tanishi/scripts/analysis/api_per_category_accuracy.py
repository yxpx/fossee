import pandas as pd
import ast
from pathlib import Path

print("="*70)
print("📊 API MODEL PER-CATEGORY ACCURACY ANALYSIS")
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

# Get all unique categories
all_cats = set()
for cats in df['manual_cats_list']:
    all_cats.update(cats)
all_cats = sorted([c for c in all_cats if c])  # Remove empty strings

print(f"\n📋 Categories: {all_cats}")

# Define models
models = [
    'anthropic_claude-sonnet-4.5',
    'deepseek_deepseek-v3.2',
    'google_gemini-2.5-flash',
    'openai_gpt-5.2',
    'openai_gpt-oss-120b',
    'qwen_qwen3-coder'
]

# Get model columns
model_cols = [c for c in df.columns if c.startswith('results_')]

print("\n" + "="*70)
print("🔍 Calculating per-category accuracy...")
print("="*70)

# Store results
category_results = []

for model in models:
    # Get all prediction columns for this model (across all runs & modes)
    model_cols_subset = [c for c in model_cols if model in c]
    
    print(f"\n📊 {model}")
    print(f"   Analyzing {len(model_cols_subset)} runs...")
    
    for category in all_cats:
        # Filter rows where manual label contains this category
        mask = df['manual_cats_list'].apply(lambda x: category in x)
        df_cat = df[mask]
        
        questions_with_category = len(df_cat)
        
        if questions_with_category == 0:
            continue
        
        # Check predictions across all runs for this model
        total_predictions = 0
        correct_predictions = 0
        
        for col in model_cols_subset:
            for idx, row in df_cat.iterrows():
                manual_cats = row['manual_cats_list']
                predicted = str(row[col]).strip()
                
                total_predictions += 1
                
                # Correct if prediction matches any manual category
                if predicted in manual_cats:
                    correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        category_results.append({
            'Model': model,
            'Category': category,
            'Questions': questions_with_category,
            'Total_Predictions': total_predictions,
            'Correct': correct_predictions,
            'Accuracy_%': round(accuracy, 2)
        })
        
        print(f"   {category}: {correct_predictions}/{total_predictions} = {accuracy:.2f}%")

# Create DataFrame
results_df = pd.DataFrame(category_results)

# Save detailed results
Path("data/exports").mkdir(parents=True, exist_ok=True)
results_df.to_csv("data/exports/api_per_category_accuracy.csv", index=False)

print("\n" + "="*70)
print("📊 SUMMARY: Model Performance by Category")
print("="*70)

# Pivot table: Models (rows) vs Categories (columns)
pivot = results_df.pivot_table(
    index='Model',
    columns='Category',
    values='Accuracy_%',
    aggfunc='mean'
).round(2)

print("\n" + str(pivot))

# Save pivot
pivot.to_csv("data/exports/api_per_category_accuracy_pivot.csv")

# Find best/worst category for each model
print("\n" + "="*70)
print("📊 Best & Worst Category per Model")
print("="*70)

for model in models:
    model_data = results_df[results_df['Model'] == model]
    if len(model_data) == 0:
        continue
    
    best = model_data.loc[model_data['Accuracy_%'].idxmax()]
    worst = model_data.loc[model_data['Accuracy_%'].idxmin()]
    
    print(f"\n{model}:")
    print(f"  ✅ Best:  {best['Category']} ({best['Accuracy_%']}%)")
    print(f"  ❌ Worst: {worst['Category']} ({worst['Accuracy_%']}%)")

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. data/exports/api_per_category_accuracy.csv (detailed)")
print("  2. data/exports/api_per_category_accuracy_pivot.csv (summary table)")
