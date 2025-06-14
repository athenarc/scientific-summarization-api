# Scientific Paper Summarization API

### Run the Application

```bash
# Development server
uvicorn summarizer_api:app --reload --host 0.0.0.0 --port 8000

# Production server
./gunicorn.sh

# Stop production server
./stop_server.sh

# Check server health
./health_check.sh
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/healthlication that generates AI-powered summaries of scientific papers using large language models. The system analyzes paper abstracts and creates coherent, well-structured summaries with proper academic citations.

## 🔬 What This System Does

This API service takes collections of scientific papers (titles and abstracts) and generates comprehensive summaries using advanced AI models. Key features include:

- **Intelligent Summarization**: Multiple summarization strategies for different use cases
- **Academic Citations**: Proper citation formatting with numbered references
- **Flexible AI Models**: Support for OpenAI, DeepSeek, local models, and other OpenAI-compatible APIs
- **Production Ready**: Comprehensive logging, error handling, validation, and monitoring
- **Scalable Architecture**: Designed for high-throughput research applications

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- AI API access (OpenAI, DeepSeek, or local model server)

### Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd bip-scientific-summarization

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env
# Edit .env with your API credentials
```

### Basic Configuration

Create a `.env` file:

```env
# Required Configuration
OPENAI_API_HOST=https://api.openai.com
OPENAI_API_KEY=your-api-key-here
MODEL=gpt-3.5-turbo

# Optional Configuration
MAX_TOKENS=1000
TEMPERATURE=0.7
TOP_P=0.95
MAX_PAPERS=50
LOG_LEVEL=INFO
```

### Run the Application

```bash
# Development server
uvicorn summarizer_api:app --reload --host 0.0.0.0 --port 8000

# Production server (recommended)
chmod +x gunicorn.sh
./gunicorn.sh
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage Examples

### Python Client

```python
import requests

# Prepare scientific papers data
papers = [
    {
        "id": "1",
        "title": "Machine Learning Applications in Personalized Medicine",
        "abstract": "This study explores the integration of machine learning algorithms in personalized medicine approaches. We developed a predictive model using patient genomic data to optimize treatment selection, achieving 85% accuracy in treatment outcome prediction."
    },
    {
        "id": "2", 
        "title": "Ethical Frameworks for AI in Healthcare Decision Making",
        "abstract": "As artificial intelligence systems become integral to clinical decision-making, establishing robust ethical frameworks becomes critical. This paper proposes a comprehensive ethical evaluation system for AI-assisted medical diagnosis."
    }
]

# Generate summary
response = requests.post("http://localhost:8000/summarize/", json={
    "papers": papers,
    "topic_name": "AI in Personalized Healthcare",
    "prompt_key": "two_paragraph"
})

result = response.json()
print(f"Topic: {result['topic_name']}")
print(f"Summary: {result['summary']}")
print(f"References: {len(result['references'])} papers cited")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {
        "id": "1",
        "title": "Deep Learning for Medical Image Analysis",
        "abstract": "We present a novel deep learning approach for automated medical image analysis, achieving state-of-the-art performance on multiple diagnostic tasks with 92% accuracy."
      }
    ],
    "topic_name": "Medical AI Diagnostics",
    "prompt_key": "single_paragraph"
  }'
```

## 🎯 API Endpoints

### POST `/summarize/`
Generate an AI-powered summary from scientific papers.

**Request:**
```json
{
  "papers": [
    {
      "id": "string|number",
      "title": "Paper title (1-500 chars)",
      "abstract": "Paper abstract (10-5000 chars)"
    }
  ],
  "topic_name": "Research Topic Name",
  "prompt_key": "single_paragraph"
}
```

**Response:**
```json
{
  "topic_name": "Research Topic Name",
  "summary": "Generated summary with citations [1], [2]...",
  "references": [
    {"id": "1", "title": "Paper Title"}
  ],
  "tokens_used": {
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350
  },
  "prompt_used": "single_paragraph",
  "processing_time_seconds": 2.45
}
```

### GET `/health`
Service health check with AI model validation.

### GET `/prompts`
List all available summarization strategies.

### GET `/docs`
Interactive API documentation (Swagger UI).

## 📝 Summarization Strategies

The system offers five distinct summarization approaches:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `comprehensive_300` | Detailed 300-word analysis with full context | In-depth research reviews |
| `strict_300` | Concise summaries with hard 300-word limit | Standardized reports |
| `strict_250` | 250-word summaries in 2-3 paragraphs | Executive summaries |
| `two_paragraph` | Methodology overview + key findings | Research presentations |
| `single_paragraph` | Focused narrative-style summary | Quick overviews |

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_HOST` | AI API endpoint URL | - | ✅ |
| `OPENAI_API_KEY` | API authentication key | - | Conditional* |
| `MODEL` | AI model identifier | - | ✅ |
| `MAX_TOKENS` | Maximum response tokens | 1000 | ❌ |
| `TEMPERATURE` | Model creativity (0.0-2.0) | 0.7 | ❌ |
| `TOP_P` | Model focus (0.0-1.0) | 0.95 | ❌ |
| `MAX_PAPERS` | Maximum papers per request | 50 | ❌ |
| `REQUEST_TIMEOUT` | API timeout in seconds | 300 | ❌ |
| `LOG_LEVEL` | Logging verbosity | INFO | ❌ |
| `CORS_ORIGINS` | Allowed CORS origins | * | ❌ |
| `ALLOWED_HOSTS` | Trusted host domains | * | ❌ |

*API key not required for local/self-hosted models

### Supported AI Providers

- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo models
- **DeepSeek**: DeepSeek-Chat models  
- **Local Models**: Text Generation Inference (TGI) servers
- **Custom**: Any OpenAI-compatible API endpoint

### Custom Prompts

Add new summarization strategies to `system_prompts.yaml`:

```yaml
prompts:
  custom_brief:
    name: "Custom Brief Summary"
    description: "Ultra-concise summaries for rapid scanning"
    content: |
      Create a brief summary focusing only on the most critical findings...
```

## � Deployment

## 🚀 Deployment

### Production Deployment with Gunicorn

The application includes production-ready deployment scripts with optimized Gunicorn configuration:

```bash
# Start the production server
./gunicorn.sh

# Check server health
./health_check.sh

# Stop the server
./stop_server.sh
```

### Deployment Scripts

#### `gunicorn.sh` - Start Server
- Configures 4 worker processes for concurrent requests
- Uses optimized Uvicorn workers for async performance
- Includes connection limits, request limits, and timeouts
- Creates PID file for process management
- Logs access and errors to separate files

#### `stop_server.sh` - Stop Server
- Gracefully stops the Gunicorn server using PID file
- Falls back to process search if PID file is missing
- Handles force termination if graceful shutdown fails

#### `health_check.sh` - Health Monitoring
- Performs HTTP health check on the `/health` endpoint
- Returns appropriate exit codes for monitoring systems

### Process Management

```bash
# View real-time logs
tail -f ./logs/summarizer_api_access.log
tail -f ./logs/summarizer_api_error.log

# Check process status
ps aux | grep gunicorn

# Restart the service
./stop_server.sh && ./gunicorn.sh

# Monitor with health checks
watch -n 5 ./health_check.sh
```

### Advanced Configuration

The Gunicorn configuration includes optimized settings:

- **Workers**: 4 processes for CPU-bound tasks
- **Worker Connections**: 1000 concurrent connections per worker
- **Request Limits**: 1000 requests per worker with jitter
- **Timeouts**: 300-second timeout for long-running summarization
- **Keep-Alive**: 2-second connection keep-alive

## 🔍 Monitoring and Logging

### Health Monitoring

```bash
# Check service health
curl http://localhost:8000/health

# Get available prompts
curl http://localhost:8000/prompts
```

### Log Files

- `./logs/summarizer_api.log` - Application logs
- `./logs/summarizer_api_access.log` - HTTP access logs (production)
- `./logs/summarizer_api_error.log` - Error logs (production)

### Metrics

The API tracks:
- Request/response times
- Token usage statistics
- Error rates and types
- AI model performance

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Ensure API is running
uvicorn summarizer_api:app --reload &

# Run tests
python test_api.py

# Stop the server
pkill -f uvicorn
```

Test coverage includes:
- Health check validation
- All summarization strategies
- Error handling scenarios
- Input validation
- AI model connectivity

## 🔧 Development

### Project Structure

```
bip-scientific-summarization/
├── summarizer_api.py          # Main FastAPI application
├── system_prompts.yaml        # Summarization strategies
├── test_api.py               # Comprehensive test suite
├── requirements.txt          # Python dependencies
├── gunicorn.sh              # Production server script
├── .env.example            # Environment template
├── logs/                   # Application logs
└── data-api-samples/       # Sample data for testing
```

### Code Quality

The codebase follows modern Python best practices:

- **Type Safety**: Full type hints with Pydantic validation
- **Error Handling**: Comprehensive exception handling with structured responses
- **Logging**: Configurable logging with proper levels
- **Security**: CORS configuration, trusted hosts, input validation
- **Performance**: Request/response timing, efficient AI client usage
- **Documentation**: OpenAPI specs with examples and descriptions

### Adding Features

1. **New Summarization Strategy**: Add to `system_prompts.yaml`
2. **Custom Middleware**: Add to the FastAPI app setup
3. **Additional Endpoints**: Follow the existing pattern with proper typing
4. **Enhanced Validation**: Extend Pydantic models

## 🤝 Contributing

1. **Code Style**: Follow PEP 8 with Black formatting
2. **Type Hints**: Add type annotations to all functions
3. **Testing**: Update test suite for new features
4. **Documentation**: Update OpenAPI specs and README
5. **Logging**: Add appropriate log statements

## 📄 License

[Specify your license here]

## 🆘 Troubleshooting

### Common Issues

**Connection Errors**
```bash
# Check API connectivity
curl -v http://localhost:8000/health
```

**Model Access Issues**
```bash
# Verify environment variables
echo $OPENAI_API_HOST $MODEL
```

**High Memory Usage**
- Reduce `MAX_TOKENS` setting
- Limit concurrent requests with Gunicorn workers
- Monitor server resources with `htop` or `ps`

**Slow Response Times**
- Check AI model performance
- Verify network connectivity
- Review `REQUEST_TIMEOUT` setting
- Consider reducing `TEMPERATURE` for faster responses

### Support

For technical support:
1. Check service health endpoints
2. Review application logs
3. Validate environment configuration
4. Test with provided sample data

---

## 🎓 Research Applications

This API is ideal for:

- **Literature Reviews**: Automated summarization of research papers
- **Meta-Analysis**: Synthesis of findings across multiple studies
- **Research Monitoring**: Tracking developments in specific fields
- **Academic Writing**: Background research and citation management
- **Grant Applications**: Summarizing related work and positioning research

The system has been designed specifically for academic and research use cases, ensuring high-quality outputs suitable for scholarly work.