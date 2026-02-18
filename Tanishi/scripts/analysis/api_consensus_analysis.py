import pandas as pd
import ast
from pathlib import Path
from collections import Counter

print("="*70)
print("📊 CONSENSUS ANALYSIS - Majority Voting (Run1 Only)")
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
    
    # Get run1 columns for this mode
    run1_cols = [f"results_run1_{mode}_{model}" for model in models]
    
    # Analyze consensus for each question
    unanimous_correct = 0
    majority_correct = 0
    no_consensus = 0
    unanimous_wrong = 0
    
    consensus_predictions = []
    
    for idx, row in df.iterrows():
        manual = row['manual_cats_list']
        
        if not manual:
            continue
        
        # Get all predictions for this question
        predictions = [str(row[col]).strip() for col in run1_cols if col in df.columns]
        
        # Count votes
        vote_count = Counter(predictions)
        most_common = vote_count.most_common(1)[0]
        consensus_pred = most_common[0]
        votes = most_common[1]
        
        # Check if consensus is correct
        consensus_correct = consensus_pred in manual
        
        # Count how many models got it right individually
        individual_correct = sum(1 for p in predictions if p in manual)
        
        # Categorize
        if votes == 6:  # Unanimous
            if consensus_correct:
                unanimous_correct += 1
            else:
                unanimous_wrong += 1
        elif votes >= 4:  # Majority (4, 5 votes)
            if consensus_correct:
                majority_correct += 1
            else:
                no_consensus += 1
        else:  # No clear consensus (tie or low votes)
            no_consensus += 1
        
        consensus_predictions.append({
            'Question': row['Sr. no.'],
            'Manual': ','.join(manual),
            'Consensus_Prediction': consensus_pred,
            'Votes': votes,
            'Total_Models': len(predictions),
            'Consensus_Correct': 'Yes' if consensus_correct else 'No',
            'Individual_Correct_Count': individual_correct
        })
    
    total = len([c for c in consensus_predictions])
    
    # Calculate ensemble accuracy
    ensemble_correct = sum(1 for c in consensus_predictions if c['Consensus_Correct'] == 'Yes')
    ensemble_accuracy = (ensemble_correct / total * 100) if total > 0 else 0
    
    # Calculate average individual accuracy
    avg_individual_correct = sum(c['Individual_Correct_Count'] for c in consensus_predictions) / (total * 6) * 100 if total > 0 else 0
    
    results.append({
        'Mode': mode,
        'Total_Questions': total,
        'Unanimous_Correct': unanimous_correct,
        'Majority_Correct': majority_correct,
        'No_Consensus': no_consensus,
        'Unanimous_Wrong': unanimous_wrong,
        'Ensemble_Accuracy_%': round(ensemble_accuracy, 2),
        'Avg_Individual_Accuracy_%': round(avg_individual_correct, 2),
        'Ensemble_Improvement_%': round(ensemble_accuracy - avg_individual_correct, 2)
    })
    
    print(f"\n📊 Results:")
    print(f"   Unanimous correct (6/6): {unanimous_correct}")
    print(f"   Majority correct (4-5/6): {majority_correct}")
    print(f"   No consensus/wrong: {no_consensus}")
    print(f"   Unanimous wrong: {unanimous_wrong}")
    print(f"\n   Ensemble accuracy: {ensemble_accuracy:.2f}%")
    print(f"   Avg individual accuracy: {avg_individual_correct:.2f}%")
    print(f"   Improvement: {ensemble_accuracy - avg_individual_correct:+.2f}%")
    
    # Save detailed predictions
    consensus_df = pd.DataFrame(consensus_predictions)
    Path("data/exports").mkdir(parents=True, exist_ok=True)
    consensus_df.to_csv(f"data/exports/api_consensus_{mode}_run1.csv", index=False)

# Save summary
results_df = pd.DataFrame(results)
results_df.to_csv("data/exports/api_consensus_summary.csv", index=False)

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  - data/exports/api_consensus_single_run1.csv")
print("  - data/exports/api_consensus_multi_run1.csv")
print("  - data/exports/api_consensus_summary.csv")
