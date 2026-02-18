import pandas as pd
import ast
from pathlib import Path
import numpy as np
from collections import Counter

print("="*70)
print("📊 VARIANCE ANALYSIS - Prediction Diversity (Run1 Only)")
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

# Define models
models = [
    'anthropic_claude-sonnet-4.5',
    'deepseek_deepseek-v3.2',
    'google_gemini-2.5-flash',
    'openai_gpt-5.2',
    'openai_gpt-oss-120b',
    'qwen_qwen3-coder'
]

results = []

for mode in ['single', 'multi']:
    print(f"\n" + "="*70)
    print(f"🔍 Mode: {mode.upper()}-LABEL")
    print("="*70)
    
    # Get run1 columns
    run1_cols = [f"results_run1_{mode}_{model}" for model in models if f"results_run1_{mode}_{model}" in df.columns]
    
    # Analyze variance per question
    question_variances = []
    
    for idx, row in df.iterrows():
        manual = row['manual_cats_list']
        
        if not manual:
            continue
        
        # Get all predictions
        predictions = [str(row[col]).strip() for col in run1_cols]
        
        # Count unique predictions
        unique_preds = len(set(predictions))
        
        # Calculate entropy (prediction diversity)
        pred_counts = Counter(predictions)
        total = len(predictions)
        entropy = -sum((count/total) * np.log2(count/total) for count in pred_counts.values())
        
        # Normalized entropy (0-1 scale)
        max_entropy = np.log2(total) if total > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Check if any model got it right
        any_correct = any(p in manual for p in predictions)
        
        question_variances.append({
            'Question': row['Sr. no.'],
            'Manual': ','.join(manual),
            'Unique_Predictions': unique_preds,
            'Entropy': round(entropy, 3),
            'Normalized_Entropy': round(normalized_entropy, 3),
            'Any_Correct': 'Yes' if any_correct else 'No',
            'Predictions': ', '.join(predictions)
        })
    
    # Calculate statistics
    avg_unique = np.mean([q['Unique_Predictions'] for q in question_variances])
    avg_entropy = np.mean([q['Normalized_Entropy'] for q in question_variances])
    
    high_variance_count = sum(1 for q in question_variances if q['Unique_Predictions'] >= 4)
    low_variance_count = sum(1 for q in question_variances if q['Unique_Predictions'] <= 2)
    
    print(f"\n📊 Variance Statistics:")
    print(f"   Avg unique predictions per question: {avg_unique:.2f}")
    print(f"   Avg normalized entropy: {avg_entropy:.3f}")
    print(f"   High variance questions (4+ unique): {high_variance_count}")
    print(f"   Low variance questions (≤2 unique): {low_variance_count}")
    
    results.append({
        'Mode': mode,
        'Avg_Unique_Predictions': round(avg_unique, 2),
        'Avg_Normalized_Entropy': round(avg_entropy, 3),
        'High_Variance_Count': high_variance_count,
        'Low_Variance_Count': low_variance_count
    })
    
    # Save detailed variance data
    variance_df = pd.DataFrame(question_variances)
    Path("data/exports").mkdir(parents=True, exist_ok=True)
    variance_df.to_csv(f"data/exports/api_variance_{mode}_run1.csv", index=False)
    
    # Show most controversial questions (high variance)
    print(f"\n📊 Most Controversial Questions (High Disagreement):")
    controversial = variance_df.nlargest(5, 'Unique_Predictions')
    for _, q in controversial.iterrows():
        print(f"\n   Q{q['Question']}: Manual={q['Manual']}, Unique Preds={q['Unique_Predictions']}")
        print(f"      Predictions: {q['Predictions']}")

# Save summary
results_df = pd.DataFrame(results)
results_df.to_csv("data/exports/api_variance_summary.csv", index=False)

print("\n" + "="*70)
print("📊 VARIANCE SUMMARY")
print("="*70)
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  - data/exports/api_variance_single_run1.csv")
print("  - data/exports/api_variance_multi_run1.csv")
print("  - data/exports/api_variance_summary.csv")
