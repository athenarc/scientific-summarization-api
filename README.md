# Scientific Summarization Demo

This repository contains code and data for a demonstration of scientific paper summarization using large language models.

## Overview

This project showcases how to leverage the capabilities of large language models to create concise, accurate summaries of scientific papers. The implementation supports both the Hugging Face Transformers library (focusing on the Qwen2.5 model series) and OpenAI-compatible API interfaces through a Text Generation Inference (TGI) server.

## Repository Contents

- **scientific-summarization-demo.ipynb**: The main Jupyter notebook for interactive experimentation
- **runner.py**: Command-line interface for batch processing summarization tasks
- **util_functions.py**: Core functionality for data processing and summary generation
- **tgi-script.sh**: Shell script to launch a TGI server with Docker
- **data.json**: JSON file containing papers on two topics: "artificial intelligence cyber security" and "data visualization"
- **data.1.json**: JSON file focused on "artificial intelligence cyber security" papers
- **data.2.json**: JSON file focused on "data visualization" papers
- **system_prompts.yaml**: YAML file containing various system prompts for different summarization strategies
- **formatted_papers.txt**: Text file with formatted paper data for easier reading

## Features

The demo implements several key features:

1. **Multiple data sources**:
   - Load papers from local JSON files
   - Search and retrieve papers via API using keywords
   
2. **Flexible model usage**:
   - HuggingFace Transformers models for local processing
   - OpenAI API-compatible interface for TGI
   
3. **Configuration of various summarization approaches** via system prompts:
   - Comprehensive 300-word summaries
   - Strict 300-word summaries
   - Strict 250-word summaries
   - Two-paragraph summaries
   - Single-paragraph summaries
   
4. **Quality control features**:
   - Automatic rewriting of summaries that exceed word limits
   - Statistical analysis of generated summaries, including token counts
   - Pretty printing of summaries with proper formatting

## Usage

### Jupyter Notebook

To run the demo interactively:

1. Load the notebook in a Jupyter environment
2. Execute the cells sequentially
3. Experiment with different models, prompts, and datasets

### Command Line Interface

The repository provides a command-line interface for batch processing:

```bash
python runner.py --request-origin [api|file] --input [keywords|filename] [options]
```

Options:
- `--request-origin`: Source of papers (`api` or `file`)
- `--input`: Keywords for API search or filename for local data
- `--response-only`: Show only the summary without statistics
- `--show-papers`: Display the full paper information
- `--word-limit`: Maximum word count for summary (default: 250)

### Docker-based TGI Server

Launch a local [TGI](https://huggingface.co/docs/text-generation-inference/index) server with the provided script:

```bash
./tgi-script.sh
```

## Core Functions

The primary function to generate summaries is `get_summary()`, which supports multiple parameters:

```python
get_summary(
    system_prompt,      # Prompt template from system_prompts.yaml
    client,             # OpenAI-compatible client
    data_input,         # Keywords or filename
    request_origin,     # "api" or "file"
    model,              # Model identifier
    response_only=False, # Whether to show only the summary
    print_response=True, # Whether to print the summary
    show_papers=False,   # Whether to show paper details
    word_limit=250       # Maximum word count
)
```

## Dependencies

* transformers
* torch
* openai
* pprint
* textwrap
* yaml
* requests
* python-dotenv
* argparse

## Models

The demo is configured to use Qwen's large language models:

* Qwen/Qwen2.5-14B-Instruct
* Qwen/Qwen2.5-14B-Instruct-1M
* Qwen/Qwen2.5-32B-Instruct

Any model compatible with Text Generation Inference can also be used.

## Example

Using the command-line interface:

```bash
python runner.py --request-origin file --input data.2.json --word-limit 300
```

Using the Python API:

```python
summary = get_summary(
    system_prompt=prompts['prompts']['single_paragraph']['content'],
    client=client,
    request_origin='api',
    data_input="machine learning medicine",
    word_limit=250
)
```