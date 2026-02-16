import json
import requests
import time
import os
import argparse
import re

def benchmark_models():
    # Configuration
    api_key = "api-key"
    available_models = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5.2",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-coder",
        "openai/gpt-oss-120b"
    ]

    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark LLM models on Yaksh questions")
    parser.add_argument("--models", nargs="*", default=[], 
                        help=f"Specific models to run. Options: {', '.join(available_models)}")
    parser.add_argument("--clean", action="store_true", help="Clear previous results before running")
    args = parser.parse_args()

    models = args.models if args.models else available_models
    
    # Validate models
    for m in models:
        if m not in available_models:
            print(f"Warning: Model '{m}' is not in the default list, but proceeding anyway.")

    print(f"Running benchmark for models: {models}")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "yaksh100.json")
    results_dir = os.path.join(base_dir, "results")
    results_single_dir = os.path.join(results_dir, "single")
    results_multi_dir = os.path.join(results_dir, "multi")
    for d in [results_dir, results_single_dir, results_multi_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Clean previous result files if requested
    if args.clean:
        print("Cleaning previous results...")
        for model in models:
            safe_model_name = model.replace("/", "_").replace(":", "_")
            model_file_single = os.path.join(results_single_dir, f"{safe_model_name}_single.txt")
            model_file_multi = os.path.join(results_multi_dir, f"{safe_model_name}_multi.txt")
            for model_file in [model_file_single, model_file_multi]:
                if os.path.exists(model_file):
                    os.remove(model_file)
                # re-create empty
                open(model_file, 'w', encoding='utf-8').close()

    # Load already processed question IDs per model and mode
    processed_map = {m: {"single": set(), "multi": set()} for m in models}
    if not args.clean:
        for model in models:
            safe_model_name = model.replace("/", "_").replace(":", "_")
            model_file_single = os.path.join(results_single_dir, f"{safe_model_name}_single.txt")
            model_file_multi = os.path.join(results_multi_dir, f"{safe_model_name}_multi.txt")
            for mode, model_file in [("single", model_file_single), ("multi", model_file_multi)]:
                if os.path.exists(model_file):
                    with open(model_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Look for IDs in old verbose format
                        found_ids = re.findall(r"Question ID: (\d+)", content)
                        # Look for IDs in new simple format (start of line: "123 A")
                        found_ids += re.findall(r"^(\d+)\s", content, re.MULTILINE)
                        processed_map[model][mode] = set(int(pid) for pid in found_ids)
                if processed_map[model][mode]:
                    print(f"Model {model} ({mode}): Resuming. {len(processed_map[model][mode])} questions already processed.")

    # Load prompt templates (single-label and multi-label) from prompt.txt
    prompt_path = os.path.join(base_dir, "prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_source = f.read()
        prompt_namespace = {}
        exec(prompt_source, {}, prompt_namespace)
        prompt_single = prompt_namespace.get("PROMPT_ITER1_SINGLE", "")
        prompt_multi = prompt_namespace.get("PROMPT_ITER_MULTI", "")
    except FileNotFoundError:
        print(f"Error: prompt.txt not found at {prompt_path}")
        return

    if not prompt_single or not prompt_multi:
        print("Error: Missing PROMPT_ITER1_SINGLE or PROMPT_ITER_MULTI in prompt.txt")
        return

    system_prompt = "You are a helpful assistant."
    
    # Read data
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: yaksh_selected_qns.json not found at {data_path}")
        return

    # Process each question
    for index, item in enumerate(questions):
        # Handle cases where "Sr. no." might be different or missing, though structure is known
        q_id = item.get("Sr. no.", index + 1)
        description = item.get("question__description", "")
        code_answer = item.get("answer", "")
        
        code_input = f"Problem Description:\n{description}\n\nCode:\n{code_answer}"
        
        print(f"Processing Question {q_id}...")
        
        for model in models:
            for mode, prompt_template in [("single", prompt_single), ("multi", prompt_multi)]:
                # Skip if already processed
                if q_id in processed_map.get(model, {}).get(mode, set()):
                    continue

                print(f"  Querying {model} ({mode})...")

                user_content = prompt_template.format(code_input=code_input)

                # Prepare request
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter sometimes
                    "X-Title": "Benchmark Script"
                }

                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "max_tokens": 2000
                }

                # Disable thinking for all except qwen3-coder
                if model != "qwen3-coder":
                    payload["include_reasoning"] = False

                output_content = ""
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )

                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [{}])
                        if choices:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                            if not content and "reasoning" in message:
                                content = message["reasoning"]
                        else:
                            content = ""

                        output_content = content
                    else:
                        print(f"    Error {response.status_code}: {response.text}")
                        output_content = f"Error: {response.status_code}\n{response.text}"
                except Exception as e:
                    print(f"    Exception: {str(e)}")
                    output_content = f"Exception: {str(e)}"

                # Save to model specific file
                safe_model_name = model.replace("/", "_").replace(":", "_")
                if mode == "single":
                    model_file = os.path.join(results_single_dir, f"{safe_model_name}_single.txt")
                else:
                    model_file = os.path.join(results_multi_dir, f"{safe_model_name}_multi.txt")

                # Extract the letters
                cleaned = output_content.strip()
                if mode == "single":
                    matches = re.findall(r'(?:^|[\s*:\-\(])((?:[A-JN]|NONE)+)(?:$|[\s*:\.\)])', cleaned, re.IGNORECASE)
                    if matches:
                        final_answer = matches[-1].upper()
                    else:
                        final_answer = cleaned.replace('\n', ' ')
                        if len(final_answer) > 100:
                            final_answer = final_answer[:100] + "..."
                else:
                    matches = re.findall(r'((?:[A-JN]|NONE)(?:,(?:[A-JN]|NONE))*)', cleaned, re.IGNORECASE)
                    if matches:
                        final_answer = matches[-1].upper()
                    else:
                        final_answer = cleaned.replace('\n', ' ')
                        if len(final_answer) > 100:
                            final_answer = final_answer[:100] + "..."

                if not final_answer:
                    final_answer = "NO_OUTPUT"

                with open(model_file, "a", encoding="utf-8") as f:
                    f.write(f"{q_id} {final_answer}\n")

                # Rate limit politeness
                time.sleep(1)

    print(f"Benchmark complete. Results saved to {results_dir}")

if __name__ == "__main__":
    benchmark_models()
