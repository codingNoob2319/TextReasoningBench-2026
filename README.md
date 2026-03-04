# LLM Reasoning Benchmark for Text Classification

## Overview
This repository contains the experimental framework and evaluation code for the research project: "Research on LLM Reasoning Methods for Text Classification." 

The project systematically evaluates and compares the performance of various Large Language Models (LLMs) across multiple text classification datasets, utilizing a diverse set of reasoning strategies to assess both base capabilities and prompt-driven logic.

## Evaluated Models

**Small-Parameter Models:**
* `qwen3-8b`
* `gpt-4o-mini`
* `gemma-3-4b-it`
* `llama-3.1-8b-instruct`

**Large-Parameter Models:**
* `deepseek-reasoner`
* `kimi-k2`
* `gpt-5`
* `gemini-2.5-flash`
* `qwen3-max`

## Reasoning Methods
* **IO (Input-Output):** Standard zero-shot/few-shot direct prompting.
* **CoT (Chain of Thought):** Step-by-step reasoning extraction.
* **ToT (Tree of Thoughts):** Multi-path exploration, critique, and resolution.
* **GoT (Graph of Thoughts):** Multi-perspective expansion and aggregation.
* **SC-CoT (Self-Consistency CoT):** Multiple sampling and majority voting.
* **BoC (Bagging of Cues):** Randomly sampling task-specific cues to force analysis from specific semantic aspects.
* **Long-CoT:** Direct invocation of the model's native thinking/reasoning capabilities without complex prompt engineering.

## Datasets
* **AG News (Subset):** Topic classification (World, Sports, Business, Sci/Tech).
* **TREC-QC:** Question classification (6 coarse-grained categories).
* **SemEval-2018 (Task 3A):** Irony detection.
* **iSarcasmEval (Task A):** Intended sarcasm detection.
* **SST-2:** Sentiment analysis.

## Quick Start

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone [https://github.com/codingNoob2319/TextReasoningBench-2026](https://github.com/codingNoob2319/TextReasoningBench-2026.git)
pip install -r requirements.txt
```

### 2. API Configuration
Create a `.env` file in the project root directory to store your API keys safely. Do not commit this file to version control.
```env
OPENAI_API_KEY=your_openai_key
DASHSCOPE_API_KEY=your_qwen_key
DEEPSEEK_API_KEY=your_deepseek_key
```

### 3. Execution
Run the evaluation suite via the command-line interface:
```bash
python main.py --model qwen3-8b --dataset ag_news --method cot --runs 1
```

**Arguments:**
* `--model`: Target model identifier.
* `--dataset`: Target dataset identifier.
* `--method`: Reasoning strategy (`io`, `cot`, `tot`, `sc_cot`, `got`, `boc`, `long_cot`).
* `--runs`: Number of independent runs for statistical stability.
* `--few_shot`: Number of few-shot examples per class (default: 0).
* `--full`: Execute on the full dataset (defaults to a 100-sample debug mode if omitted).