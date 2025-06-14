import os
import logging
import yaml
import time
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
from openai import Client
from openai.types.chat import (
    ChatCompletionMessageParam, 
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletion
)
from openai.types.completion_usage import CompletionUsage
from dotenv import load_dotenv

# Configure logging
def setup_logging():
    """Setup logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler('./logs/summarizer_api.log'),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()  # Load environment variables from .env file

# --- Configuration Class ---
class Config:
    """Application configuration class with comprehensive validation."""
    
    def __init__(self):
        self.prompts_file = os.getenv("PROMPTS_FILE", 'system_prompts.yaml')
        self.openai_api_host = os.getenv("OPENAI_API_HOST")
        self.openai_api_port = os.getenv("OPENAI_API_PORT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.95"))
        self.max_papers = int(os.getenv("MAX_PAPERS", "50"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "300"))
        self.allowed_hosts = os.getenv("ALLOWED_HOSTS", "*").split(",")
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        
        self._validate_config()
        self.base_url = self._build_base_url()
    
    def _validate_config(self):
        """Validate essential configuration variables with detailed error messages."""
        errors = []
        
        if not self.openai_api_host:
            errors.append("OPENAI_API_HOST environment variable must be set.")
        if not self.model_name:
            errors.append("MODEL environment variable must be set.")
        if self.max_tokens <= 0:
            errors.append("MAX_TOKENS must be a positive integer.")
        if not (0.0 <= self.temperature <= 2.0):
            errors.append("TEMPERATURE must be between 0.0 and 2.0.")
        if not (0.0 <= self.top_p <= 1.0):
            errors.append("TOP_P must be between 0.0 and 1.0.")
        if self.max_papers <= 0:
            errors.append("MAX_PAPERS must be a positive integer.")
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise RuntimeError(error_message)
    
    def _build_base_url(self) -> str:
        """Build the base URL for the AI client."""
        if "localhost" in self.openai_api_host or "127.0.0.1" in self.openai_api_host:
            return f"{self.openai_api_host}:{self.openai_api_port}/v1/" if self.openai_api_port else f"{self.openai_api_host}/v1/"
        else:
            return f"{self.openai_api_host}/v1/" if not self.openai_api_host.endswith("/") else f"{self.openai_api_host}v1/"

config = Config()

# --- Pydantic Models for Request and Response ---

class Paper(BaseModel):
    """Defines the structure for a single paper."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: Union[str, int] 
    title: str = Field(..., min_length=1, max_length=500, description="The title of the paper")
    abstract: str = Field(..., min_length=10, max_length=5000, description="The abstract of the paper")
    
    @field_validator('title', 'abstract')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Ensure text fields are not just whitespace."""
        if not v.strip():
            raise ValueError('Field cannot be empty or just whitespace')
        return v.strip()

class SummarizationRequest(BaseModel):
    """Defines the structure for the summarization request body."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    papers: List[Paper] = Field(
        ..., 
        min_length=1, 
        max_length=50,
        description="List of papers to summarize",
        examples=[[{
            "id": "1",
            "title": "Machine Learning in Healthcare",
            "abstract": "This paper explores machine learning applications..."
        }]]
    )
    topic_name: str = Field(
        default="Research Summary", 
        description="A descriptive name for the collection of papers being summarized", 
        max_length=200,
        examples=["AI in Healthcare", "Climate Change Research"]
    )
    prompt_key: Optional[str] = Field(
        default=None, 
        description="The prompt strategy to use from system_prompts.yaml. If not specified, automatically uses 'single_paragraph' for ≤5 papers or 'lit_review' for >5 papers",
        examples=["single_paragraph", "lit_review", "strict_250"]
    )

class ReferenceItem(BaseModel):
    """Defines the structure for a reference item in the response."""
    id: Union[str, int]
    title: str

class TokenUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class SummarizationResponse(BaseModel):
    """Defines the structure for the summarization response."""
    topic_name: str = Field(description="The topic name from the request")
    summary: str = Field(description="The generated summary with citations")
    references: List[ReferenceItem] = Field(description="List of referenced papers")
    tokens_used: Optional[TokenUsage] = Field(default=None, description="Token usage statistics")
    prompt_used: str = Field(description="The prompt strategy that was used")
    processing_time_seconds: Optional[float] = Field(default=None, description="Time taken to process the request")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    message: str = Field(description="Detailed status message")
    model: str = Field(description="AI model being used")
    timestamp: str = Field(description="Current timestamp")
    version: str = Field(description="API version") 

# --- System Prompts Management ---
class SystemPromptsManager:
    """Manages loading and access to system prompts."""
    
    def __init__(self, prompts_file: str):
        self.prompts_file = prompts_file
        self.prompts: Dict[str, Any] = {}
        self.load_prompts()
    
    def load_prompts(self):
        """Load prompts from YAML file with error handling."""
        try:
            with open(self.prompts_file, 'r') as file:
                loaded_yaml = yaml.safe_load(file)
                if isinstance(loaded_yaml, dict) and 'prompts' in loaded_yaml:
                    self.prompts = loaded_yaml['prompts']
                    logger.info(f"Loaded {len(self.prompts)} system prompts from {self.prompts_file}")
                else:
                    logger.warning(f"'{self.prompts_file}' is not structured as expected or 'prompts' key is missing.")
                    self._set_default_prompt()
        except FileNotFoundError:
            logger.warning(f"System prompts file '{self.prompts_file}' not found. Using default prompt.")
            self._set_default_prompt()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML from '{self.prompts_file}': {e}. Using default prompt.")
            self._set_default_prompt()
    
    def _set_default_prompt(self):
        """Set a default prompt when loading fails."""
        self.prompts = {
            "default_summary": {
                "content": "Provide a concise summary of the following scientific papers."
            }
        }
    
    def get_prompt_content(self, prompt_key: str) -> str:
        """Get prompt content by key with fallback."""
        if prompt_key in self.prompts and isinstance(self.prompts.get(prompt_key), dict) and 'content' in self.prompts[prompt_key]:
            return self.prompts[prompt_key]['content']
        elif self.prompts:
            first_key = next(iter(self.prompts))
            logger.warning(f"Prompt key '{prompt_key}' not found. Using fallback prompt '{first_key}'.")
            return self.prompts[first_key].get('content', "Summarize the provided texts.")
        else:
            logger.warning("No system prompts loaded. Using basic default prompt.")
            return "Provide a concise summary of the following scientific papers."

# --- Dependency Functions ---
def get_config() -> Config:
    """Dependency function to get configuration."""
    return config

def get_prompts_manager() -> SystemPromptsManager:
    """Dependency function to get prompts manager."""
    return prompts_manager

def get_ai_client() -> Client:
    """Dependency function to get AI client."""
    return ai_client

# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Scientific Paper Summarization API")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Base URL: {config.base_url}")
    logger.info(f"Max papers per request: {config.max_papers}")
    
    # Validate configuration on startup
    try:
        test_messages = [
            ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
            ChatCompletionUserMessageParam(role="user", content="Say 'OK' if you can respond.")
        ]
        response, _ = generate_ai_response(ai_client, test_messages, config.model_name)
        logger.info("AI client validated successfully")
    except Exception as e:
        logger.warning(f"AI client validation failed: {e}")
    
    yield
    # Shutdown
    logger.info("Shutting down Scientific Paper Summarization API")

# --- Global Variables & Initialization ---
config = Config()
prompts_manager = SystemPromptsManager(config.prompts_file)

app = FastAPI(
    title="Scientific Paper Summarization API",
    description="An API that receives a list of scientific papers and returns an AI-generated summary using large language models.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add security middleware
if config.allowed_hosts != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=config.allowed_hosts
    )

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# Initialize AI client
ai_client = Client(
    base_url=config.base_url,
    api_key=config.openai_api_key if config.openai_api_key else "not_needed",
    timeout=config.request_timeout
)

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured responses."""
    logger.warning(f"HTTP {exc.status_code} error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with proper logging."""
    logger.error(f"Unexpected error in {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_server_error",
                "status_code": 500,
                "message": "An unexpected error occurred. Please try again later.",
                "timestamp": time.time()
            }
        }
    )

# --- Helper Functions ---

def format_papers_for_prompt(papers: List[Paper]) -> str:
    """Formats a list of Paper objects into a string for the AI prompt."""
    formatted_output = "\nPapers to analyze:\n\n"
    for paper in papers:
        formatted_output += f"Paper ID: {paper.id}\n"
        formatted_output += f"Title: {paper.title}\n"
        formatted_output += f"Abstract: {paper.abstract}\n"
        formatted_output += "-" * 80 + "\n\n"
    return formatted_output

def create_ai_messages(system_prompt_content: str, formatted_papers: str) -> List[ChatCompletionMessageParam]:
    """Creates the message structure for the AI model using Pydantic models."""
    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt_content),
        ChatCompletionUserMessageParam(role="user", content=formatted_papers)
    ]
    return messages

def generate_ai_response(client: Client, messages: List[ChatCompletionMessageParam], model: str) -> tuple[str, Optional[CompletionUsage]]:
    """
    Generates a response from the AI model.
    Returns a tuple of (summary_text, usage_stats).
    Raises HTTPException if summary content is not generated.
    """
    try:
        logger.debug(f"Sending request to AI model: {model}")
        response_object: ChatCompletion = client.chat.completions.create(
            model=model, 
            messages=messages, 
            stream=False,
            max_tokens=config.max_tokens, 
            temperature=config.temperature, 
            top_p=config.top_p       
        )
        
        if not response_object.choices or not response_object.choices[0].message:
            logger.error("AI response missing choices or message.")
            raise HTTPException(
                status_code=500, 
                detail="AI response malformed: missing choices or message."
            )

        summary_content = response_object.choices[0].message.content
        
        if summary_content is None or not summary_content.strip():
            logger.error("AI response content is None or empty.")
            raise HTTPException(
                status_code=500, 
                detail="AI failed to generate summary content."
            )
            
        usage_stats = response_object.usage 
        logger.debug(f"AI response generated successfully. Usage: {usage_stats}")
        return summary_content, usage_stats
        
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error during AI model call: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"AI service unavailable: {str(e)}"
        )

# --- API Endpoints ---

@app.get(
    "/health", 
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check the health status of the API and AI model connectivity"
)
async def health_check(
    config: Config = Depends(get_config),
    ai_client: Client = Depends(get_ai_client)
):
    """Health check endpoint with comprehensive validation."""
    from datetime import datetime
    
    try:
        # Test AI client connection with timeout
        test_messages = [
            ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
            ChatCompletionUserMessageParam(role="user", content="Say 'OK' if you can respond.")
        ]
        response, usage = generate_ai_response(ai_client, test_messages, config.model_name)
        
        return HealthResponse(
            status="healthy",
            message="API is operational and AI model is accessible",
            model=config.model_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"AI model not accessible: {str(e)}",
            model=config.model_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            version="1.0.0"
        )

@app.get(
    "/prompts",
    tags=["Configuration"],
    summary="List Available Prompts",
    description="Get a list of all available summarization prompt strategies"
)
async def list_available_prompts(
    prompts_manager: SystemPromptsManager = Depends(get_prompts_manager)
):
    """List all available system prompts with metadata."""
    return {
        "available_prompts": list(prompts_manager.prompts.keys()),
        "total_count": len(prompts_manager.prompts),
        "prompts": {
            key: {
                "name": value.get("name", key), 
                "description": value.get("description", "No description available")
            }
            for key, value in prompts_manager.prompts.items()
        }
    }

@app.post(
    "/summarize/", 
    response_model=SummarizationResponse,
    tags=["Summarization"],
    summary="Generate Summary",
    description="Generate an AI-powered summary from a collection of scientific papers",
    status_code=status.HTTP_200_OK
)
async def summarize_papers_endpoint(
    request_data: SummarizationRequest = Body(...),
    config: Config = Depends(get_config),
    prompts_manager: SystemPromptsManager = Depends(get_prompts_manager),
    ai_client: Client = Depends(get_ai_client)
):
    """
    Generate a comprehensive summary from scientific papers.
    
    This endpoint accepts a collection of scientific papers and generates
    a coherent summary using AI models with proper citations.
    
    Prompt selection behavior:
    - If prompt_key is not specified, automatically selects:
      * "single_paragraph" for 5 or fewer papers
      * "lit_review" for more than 5 papers
    - If prompt_key is provided, uses the specified prompt regardless of paper count
    """
    start_time = time.time()
    logger.info(f"Received summarization request for {len(request_data.papers)} papers with topic '{request_data.topic_name}'")
    
    # Validate papers count
    if not request_data.papers:
        raise HTTPException(
            status_code=400, 
            detail="No papers provided for summarization."
        )
    
    if len(request_data.papers) > config.max_papers:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many papers provided. Maximum allowed: {config.max_papers}, received: {len(request_data.papers)}"
        )

    # Automatically select prompt based on number of papers if not explicitly provided
    if request_data.prompt_key:
        selected_prompt_key = request_data.prompt_key
        logger.info(f"Using explicitly requested prompt: '{selected_prompt_key}'")
    else:
        # Use "lit_review" for more than 5 papers, "single_paragraph" for 5 or fewer
        selected_prompt_key = "lit_review" if len(request_data.papers) > 5 else "single_paragraph"
        logger.info(f"Auto-selected prompt '{selected_prompt_key}' based on {len(request_data.papers)} papers")
    
    system_prompt_content = prompts_manager.get_prompt_content(selected_prompt_key)

    # Format papers and create AI messages
    formatted_papers_string = format_papers_for_prompt(request_data.papers)
    ai_messages = create_ai_messages(system_prompt_content, formatted_papers_string)

    # Generate summary
    summary_text, usage_stats = generate_ai_response(ai_client, ai_messages, config.model_name)

    # Build response
    references_list = [ReferenceItem(id=p.id, title=p.title) for p in request_data.papers]
    
    token_info = None
    if usage_stats: 
        token_info = TokenUsage(
            prompt_tokens=usage_stats.prompt_tokens,
            completion_tokens=usage_stats.completion_tokens,
            total_tokens=usage_stats.total_tokens,
        )

    processing_time = time.time() - start_time
    logger.info(f"Successfully generated summary for topic '{request_data.topic_name}' in {processing_time:.2f}s")
    
    return SummarizationResponse(
        topic_name=request_data.topic_name,
        summary=summary_text, 
        references=references_list,
        tokens_used=token_info,
        prompt_used=selected_prompt_key,
        processing_time_seconds=round(processing_time, 3)
    )
