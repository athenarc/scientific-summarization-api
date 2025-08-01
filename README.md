# Scientific Paper Summarization API

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

A FastAPI application that generates AI-powered summaries of multiple scientific papers. This system analyzes collections of paper abstracts to produce coherent, well-structured summaries with academic citations, making it ideal for literature reviews, meta-analysis, and research monitoring.

### Key Features

-   **Intelligent Summarization**: Utilizes multiple, distinct strategies to create summaries tailored for different use cases, from quick overviews to in-depth literature reviews.
-   **Academic Citations**: Automatically formats summaries with proper, numbered in-text citations and a corresponding reference list.
-   **Flexible AI Provider Support**: Compatible with OpenAI, DeepSeek, local model servers (TGI, vLLM, Ollama), and other OpenAI-compatible APIs.
-   **Production Ready**: Built with comprehensive logging, robust error handling, health monitoring endpoints, and a scalable Gunicorn deployment script.

---

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### 1. Prerequisites

-   Python 3.11+
-   An OpenAI-compatible API, available through:
    -   A paid service (e.g., OpenAI, DeepSeek).
    -   A local model server (e.g., TGI, vLLM, Ollama).

### 2. Installation

First, clone the repository and install the required Python dependencies.

```bash
# Clone the repository
git clone git@github.com:pkoloveas/bip-scientific-summarization.git
cd bip-scientific-summarization

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file by copying the example template. This file will store your API credentials and other settings.

```bash
# Create environment configuration from the template
cp .env.example .env
```

Next, open the `.env` file and add your specific configuration.

**`.env` File Example:**

```env
# REQUIRED - API and Model Configuration
OPENAI_API_HOST=your-openai-api-host-here
OPENAI_API_KEY=your-openai-api-key-here
MODEL=your-model-name-here

# OPTIONAL - Adjust model and application behavior
MAX_TOKENS=1500
TEMPERATURE=0.7
REQUEST_TIMEOUT=300
MAX_PAPERS=50
LOG_LEVEL=INFO
```

---

## ▶️ Running the Application

You can run the server in development mode for testing or in production mode.

#### Development Server
For local development with hot-reloading enabled.

```bash
uvicorn summarizer_api:app --reload --host 0.0.0.0 --port 8000
```

#### Production Server
Uses the provided Gunicorn script for a robust, multi-worker setup.

```bash
# Make the script executable (only needs to be done once)
chmod +x gunicorn.sh

# Start the production server
./gunicorn.sh
```

Once the server is running, the following endpoints will be available:

-   **API Base URL**: `http://localhost:8000`
-   **Interactive Docs (Swagger)**: `http://localhost:8000/docs`
-   **Health Check**: `http://localhost:8000/health`

---

## 📖 Usage

Interact with the API using any HTTP client. Here are examples using cURL and Python.

### API Endpoints

| Method | Endpoint      | Description                                                 |
| :----- | :------------ | :---------------------------------------------------------- |
| `POST` | `/summarize/` | Generates a summary from a list of scientific papers.       |
| `GET`  | `/health`     | Checks the service status and AI model connectivity.        |
| `GET`  | `/prompts`    | Lists all available summarization strategies (`prompt_key`). |

### `POST /summarize/`

**Request Body:**

```json
{
  "papers": [
    {
      "id": "string | number",
      "title": "Paper Title (1-500 chars)",
      "abstract": "Paper Abstract (0-5000 chars)"
    }
  ],
  "topic_name": "Name for the Research Topic",
  "prompt_key": "single_paragraph"
}
```

-   **`papers`**: A list of objects, each containing the `id`, `title`, and `abstract` of a paper.
-   **`topic_name`**: A descriptive name for the collection of papers.
-   **`prompt_key`** (Optional): The summarization strategy to use. If omitted, the API automatically selects a strategy based on the number of papers.

**Successful Response (200 OK):**

```json
{
  "topic_name": "AI in Personalized Healthcare",
  "summary": "This is the generated summary, with citations appearing as [1] and [2]...",
  "references": [
    { "id": "1", "title": "Machine Learning Applications in Personalized Medicine" },
    { "id": "2", "title": "Ethical Frameworks for AI in Healthcare Decision Making" }
  ],
  "tokens_used": {
    "prompt_tokens": 450,
    "completion_tokens": 320,
    "total_tokens": 770
  },
  "prompt_used": "concise",
  "processing_time_seconds": 5.12
}
```

### cURL Example

Here is a basic example to get you started.

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "1",
        "title": "Deep Learning for Medical Image Analysis",
        "abstract": "We present a novel deep learning approach..."
      }
    ],
    "topic_name": "Medical AI Diagnostics"
  }'
```
*For more detailed and realistic examples, including how to generate a literature review from a larger set of papers, see the **[cURL Examples](curl_examples.md)** file.*

### Python Client Example

```python
import requests
import json

# Prepare scientific papers data
papers_data = {
    "papers": [
        {
            "id": "1",
            "title": "Machine Learning Applications in Personalized Medicine",
            "abstract": "This study explores the integration of machine learning algorithms..."
        },
        {
            "id": "2",
            "title": "Ethical Frameworks for AI in Healthcare Decision Making",
            "abstract": "As artificial intelligence systems become integral to clinical decision-making..."
        }
    ],
    "topic_name": "AI in Personalized Healthcare",
    "prompt_key": "two_paragraph"
}

# Generate summary
try:
    response = requests.post("http://localhost:8000/summarize/", json=papers_data)
    response.raise_for_status()  # Raises an exception for bad status codes
    result = response.json()
    print(f"Topic: {result['topic_name']}\n")
    print(f"Summary:\n{result['summary']}\n")
    print(f"References Cited: {len(result['references'])}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

---

## ⚙️ Advanced Configuration

### Environment Variables

The application's behavior can be fine-tuned using the following environment variables.

| Variable          | Description                                         | Default | Required    |
| :---------------- | :-------------------------------------------------- | :------ | :---------- |
| `OPENAI_API_HOST` | The base URL for the AI provider's API.             | -       | ✅          |
| `OPENAI_API_KEY`  | Your API authentication key.                        | -       | Conditional\* |
| `MODEL`           | The specific model identifier (e.g., `gpt-4-turbo`).  | -       | ✅          |
| `MAX_TOKENS`      | The maximum number of tokens to generate.           | `1000`  | ❌          |
| `TEMPERATURE`     | Model creativity (0.0 to 2.0).                      | `0.7`   | ❌          |
| `TOP_P`           | Nucleus sampling parameter (0.0 to 1.0).            | `0.95`  | ❌          |
| `MAX_PAPERS`      | Maximum number of papers allowed in a single request. | `50`    | ❌          |
| `REQUEST_TIMEOUT` | Timeout for requests to the AI provider (seconds).  | `300`   | ❌          |
| `LOG_LEVEL`       | Logging verbosity (e.g., `INFO`, `DEBUG`).          | `INFO`  | ❌          |
| `CORS_ORIGINS`    | Allowed CORS origins (comma-separated).             | `*`     | ❌          |
| `ALLOWED_HOSTS`   | Trusted host domains (comma-separated).             | `*`     | ❌          |

*\* The `OPENAI_API_KEY` is not required for local models or providers that do not use key-based authentication.*

### Summarization Strategies

The API uses different prompts to control the style and structure of the generated summary.

| `prompt_key`       | Description                                              | Best For                       |
| :----------------- | :------------------------------------------------------- | :----------------------------- |
| `concise` | A focused, narrative-style summary.                      | Quick overviews.               |
| `two_paragraph`    | A summary split into methodology and key findings.       | Research presentations.        |
| `lit_review`       | A 3-4 paragraph literature review (approx. 400-500 words). | Academic literature synthesis. |

#### Automatic Prompt Selection
If you do not provide a `prompt_key` in your request, the API will automatically select one based on the number of papers submitted:
-   **1-5 papers**: Uses `concise` for a short summary.
-   **6+ papers**: Uses `lit_review` for a more comprehensive synthesis.

#### Custom Prompts
You can add your own summarization strategies by editing the `system_prompts.yaml` file. Simply follow the existing format to define a new prompt.

---

## 📦 Deployment & Monitoring

### Production Deployment
The included scripts are configured for a production-ready deployment using Gunicorn.

```bash
# Start the production server in the background
./gunicorn.sh

# Check the server's health
./health_check.sh

# Stop the server gracefully
./stop_server.sh
```

The `gunicorn.sh` script is optimized for performance, creating multiple worker processes to handle concurrent requests and logging all access and error events to the `./logs/` directory.

### Monitoring
Check the application's health and view real-time logs.

```bash
# Check process status
ps aux | grep gunicorn

# View real-time access and error logs
tail -f ./logs/summarizer_api_access.log
tail -f ./logs/summarizer_api_error.log
```

---

## 🧪 Testing

To run the test suite, start the development server in one terminal, then run the tests in another.

```bash
# Terminal 1: Start the server
uvicorn summarizer_api:app --reload

# Terminal 2: Run the tests
python test_api.py
```
The test suite covers all primary API functionality, including all summarization strategies, input validation, and error handling scenarios.

---

## 📄 License

This project is licensed under the GPL-2.0 License. See the [LICENSE](LICENSE) file for more details.
