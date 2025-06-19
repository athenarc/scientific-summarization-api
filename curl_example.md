# cURL Examples for the Scientific Paper Summarization API

This document provides practical `curl` examples to demonstrate the core features of the API.

*These examples use `jq` to pretty-print the JSON output. You can install it via `brew install jq` or `sudo apt-get install jq`, or simply remove `| jq` from the commands.*

---

## 1. Checking API Status

Before making requests, you can check the service's health and see which summarization prompts are available.

#### Health Check
Verify that the API is running and can connect to the AI model.

```bash
curl "http://localhost:8000/health" | jq
```

#### List Available Prompts
Get a list of all supported summarization strategies (`prompt_key` values).

```bash
curl "http://localhost:8000/prompts" | jq
```

---

## 2. Generating Summaries

The primary endpoint is `/summarize/`. The API automatically chooses the best summarization strategy based on the number of papers you provide.

### Example A: Quick Summary (Fewer than 6 Papers)

When you provide a small number of papers, the API defaults to generating a concise, single-paragraph summary.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "med_img_01",
        "title": "Deep Learning for Medical Image Analysis",
        "abstract": "We present a novel deep learning approach for automated medical image analysis, achieving state-of-the-art performance on multiple diagnostic tasks with 92% accuracy."
      },
      {
        "id": "ethics_01",
        "title": "Ethical Considerations in AI-Assisted Diagnosis",
        "abstract": "This paper examines the ethical implications of using artificial intelligence in medical diagnosis, proposing guidelines for responsible implementation in clinical settings."
      },
      {
        "id": "federated_01",
        "title": "Federated Learning for Privacy-Preserving Medical AI",
        "abstract": "We introduce a federated learning framework that enables collaborative training of medical AI models while preserving patient privacy and complying with healthcare regulations."
      }
    ],
    "topic_name": "AI in Medical Diagnostics"
  }' | jq
```

### Example B: Detailed Literature Review (6+ Papers)

When you provide six or more papers, the API automatically switches to the `lit_review` prompt to generate a more comprehensive, multi-paragraph synthesis suitable for academic work.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "fin_ml_01",
        "title": "Machine Learning in Finance: A Survey",
        "abstract": "This paper provides a comprehensive survey of machine learning algorithms applied to algorithmic trading, portfolio management, and risk assessment, highlighting key trends and challenges."
      },
      {
        "id": "fin_dl_02",
        "title": "Deep Learning for Credit Scoring and Default Prediction",
        "abstract": "Our research shows that deep neural networks can improve the accuracy of credit risk evaluation by up to 15% compared to traditional logistic regression models."
      },
      {
        "id": "fin_nlp_03",
        "title": "Sentiment Analysis of Financial News Using Natural Language Processing",
        "abstract": "We developed NLP techniques to extract market sentiment from financial news articles, demonstrating a strong correlation between sentiment scores and subsequent market movements."
      },
      {
        "id": "fin_block_04",
        "title": "Applications of Blockchain Technology in Modern Banking",
        "abstract": "This study explores the application of blockchain and distributed ledger technology for creating secure, transparent, and efficient financial transaction systems."
      },
      {
        "id": "fin_quant_05",
        "title": "The Threat of Quantum Computing to Financial Cryptography",
        "abstract": "An analysis of how quantum algorithms like Shor’s algorithm threaten the security of current encryption methods used to protect financial systems and digital assets."
      },
      {
        "id": "fin_eth_06",
        "title": "A Framework for Ethical AI in Financial Decision Making",
        "abstract": "We examine the ethical implications of AI-driven financial products and propose a new framework for ensuring fairness, transparency, and accountability in automated financial services."
      }
    ],
    "topic_name": "The Impact of AI and Technology on Modern Finance"
  }' | jq
```

---

## 3. Manually Selecting a Strategy

You can override the automatic selection by explicitly setting a `prompt_key`. This is useful if you want a literature review for a small number of papers, or a brief summary for a large set.

This example forces the use of the `two_paragraph` strategy for only two papers.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "neuro_01",
        "title": "Neural Networks for Advanced Time Series Prediction",
        "abstract": "We propose a novel neural network architecture for time series forecasting that effectively combines LSTM layers with attention mechanisms to capture complex temporal dependencies."
      },
      {
        "id": "dl_comp_02",
        "title": "A Comparative Analysis of Deep Learning Models for Sequential Data",
        "abstract": "This study performs a comparative analysis of various deep learning approaches, including RNNs, LSTMs, and Transformers, for sequential data processing to identify optimal architectures for different tasks."
      }
    ],
    "topic_name": "Deep Learning for Time Series",
    "prompt_key": "two_paragraph"
  }' | jq
