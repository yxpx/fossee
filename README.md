## Project Overview

This repository hosts a comparative benchmark analysis of six Large Language Models (LLMs) in detecting logical, syntax, and semantic errors within Python code snippets. The study evaluates proprietary models (GPT-5.2, Claude Sonnet 4.5, Gemini 2.5 Flash) and open-weights models (DeepSeek v3.2, Qwen3-Coder, GPT-OSS 120b) across Single-Label and Multi-Label classification tasks.

## Repository Structure

```text
├───analysis                   # Analytical outputs
│       benchmark_analysis.ipynb   # Jupyter notebook for statistical analysis and plotting
│       benchmark_analysis.pdf     # Final PDF report of findings
│
├───data                       # Input assets
│       classification.pdf         # Error taxonomy and classification guidelines
│       prompt.txt                 # System prompts used for LLM inference
│       yaksh100.json              # Dataset of 100 Python code snippets
│
├───results                    # Generated outputs
│   ├───csv                    # Aggregated data used for analysis (Ground Truth & Model Predictions)
│   │       ground.csv
│   │       yaksh_multi.csv
│   │       yaksh_single.csv
│   │
│   ├───results_first_iteration    # Raw model outputs (Run 1)
│   │   ├───multi                  # Multi-label inference logs (.txt)
│   │   └───single                 # Single-label inference logs (.txt)
│   │
│   └───results_second_iteration    # Raw model outputs (Run 2)
│       ├───multi
│       └───single
│
└───scripts                    # Execution utilities
        benchmark_yaksh.py         # Script for running LLM inference or data aggregation

```

## Methodology

The analysis evaluates model performance against human-annotated Ground Truth (GT) using a multi-metric approach:

* **Case A (Precision):** Measures if predictions are strict subsets of the GT (minimizing hallucinations).
* **Case B (Recall):** Measures if the GT is fully contained within predictions (minimizing missed errors).
* **Case D (Jaccard Index):** A strict accuracy metric () penalizing both hallucinations and omissions.
* **Consensus Analysis:** Quantifies inter-model agreement rates to assess stability across iterations.

## Key Findings

1. **Consensus Collapse:** Model agreement degrades by approximately 50% when moving from restricted (Single) to open-ended (Multi) tasks.
2. **Performance Leader:** GPT-5.2 demonstrated the highest reliability (56.8% Jaccard Score) in complex multi-error scenarios.
3. **Behavioral Divergence:** Models exhibit distinct profiles: Conservative models (e.g., Gemini) prioritize precision, while Aggressive models (e.g., Qwen/DeepSeek) prioritize recall at the cost of higher false positive rates.
4. **Bias Detection:** A significant bias toward detecting Logical Errors (Category J) was observed across 5 of the 6 models tested.

```