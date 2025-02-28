# Scientific Summarization Demo

This repository contains code and data for a demonstration of scientific paper summarization using large language models.

## Overview

This project showcases how to leverage the capabilities of large language models to create concise, accurate summaries of scientific papers. The implementation uses the Hugging Face Transformers library with a focus on the Qwen2.5 model series.

## Repository Contents

- **scientific-summarization-demo.ipynb**: The main Jupyter notebook that contains the code implementation
- **data.json**: JSON file containing papers on two topics: "artificial intelligence cyber security" and "data visualization"
- **data.1.json**: JSON file focused on "artificial intelligence cyber security" papers
- **data.2.json**: JSON file focused on "data visualization" papers
- **system_prompts.yaml**: YAML file containing various system prompts for different summarization strategies
- **formatted_papers.txt**: Text file with formatted paper data for easier reading

## Features

The demo implements several key features:

1. **Loading and processing of scientific papers** from JSON files
2. **Configuration of various summarization approaches** via system prompts:
   - Comprehensive 300-word summaries
   - Strict 300-word summaries
   - Strict 250-word summaries
   - Two-paragraph summaries
   - Single-paragraph summaries
3. **Automatic rewriting** of summaries that exceed word limits
4. **Statistical analysis** of generated summaries, including token counts
5. **Pretty printing** of summaries with proper formatting

## Usage

To run the demo:

1. Load the notebook in a Jupyter environment
2. Execute the cells sequentially
3. Experiment with different models, prompts, and datasets

The primary function to generate summaries is `get_summary()`, which takes the following parameters:

```python
get_summary(data_file, system_prompt, model, tokenizer, response_only=False, print_response=True, show_papers=False)
```

### Dependencies
* transformers
* torch
* json
* pprint
* textwrap
* yaml

### Models

The demo is configured to use Qwen's large language models:

* Qwen/Qwen2.5-14B-Instruct
* Qwen/Qwen2.5-14B-Instruct-1M
* Qwen/Qwen2.5-32B-Instruct

### Example

```python
data_file = "data.2.json"
system_prompt = prompts['prompts']['single_paragraph']['content']
summary = get_summary(data_file, system_prompt, model, tokenizer)
```

This will generate a single-paragraph summary of the papers in the "data visualization" topic from data.2.json.