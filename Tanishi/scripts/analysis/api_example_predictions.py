import pandas as pd
import ast
from pathlib import Path

print("="*70)
print("📊 EXAMPLE PREDICTIONS (Correct & Wrong Cases)")
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

# Use best model
best_model = 'openai_gpt-5.2'
# Get one prediction column (e.g., run1_single)
example_col = f"results_run1_single_{best_model}"

if example_col not in df.columns:
    # Fallback to first available column
    model_cols = [c for c in df.columns if best_model in c]
    example_col = model_cols[0] if model_cols else None

if not example_col:
    print("❌ No prediction column found!")
    exit()

print(f"\n🔍 Using column: {example_col}")

# Find correct and wrong examples
correct_examples = []
wrong_examples = []

for idx, row in df.iterrows():
    manual = row['manual_cats_list']
    
    if not manual:
        continue
    
    pred = str(row[example_col]).strip()
    question_num = row['Sr. no.']
    question_text = str(row['question__description'])[:150] + "..."  # Truncate
    
    example = {
        'Question_Number': question_num,
        'Question_Text': question_text,
        'Manual_Label': ','.join(manual),
        'Model_Prediction': pred,
        'Match': 'Correct' if pred in manual else 'Wrong'
    }
    
    if pred in manual:
        correct_examples.append(example)
    else:
        wrong_examples.append(example)

print(f"\n✅ Found {len(correct_examples)} correct predictions")
print(f"❌ Found {len(wrong_examples)} wrong predictions")

# Select examples (5 correct, 5 wrong)
selected_correct = correct_examples[:5]
selected_wrong = wrong_examples[:5]

# Create combined DataFrame
examples_df = pd.DataFrame(selected_correct + selected_wrong)

# Save
Path("data/exports").mkdir(parents=True, exist_ok=True)
examples_df.to_csv("data/exports/api_example_predictions.csv", index=False)

print("\n" + "="*70)
print("📊 CORRECT PREDICTION EXAMPLES")
print("="*70)

for ex in selected_correct:
    print(f"\nQuestion {ex['Question_Number']}:")
    print(f"  Manual: {ex['Manual_Label']}")
    print(f"  Predicted: {ex['Model_Prediction']} ✓")

print("\n" + "="*70)
print("📊 WRONG PREDICTION EXAMPLES")
print("="*70)

for ex in selected_wrong:
    print(f"\nQuestion {ex['Question_Number']}:")
    print(f"  Manual: {ex['Manual_Label']}")
    print(f"  Predicted: {ex['Model_Prediction']} ✗")

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated: data/exports/api_example_predictions.csv")
