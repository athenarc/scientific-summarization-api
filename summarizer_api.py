import os
import yaml
import json # For potential future use, though FastAPI handles JSON well
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from openai import Client # Assuming 'openai' library is installed
# Import specific message types for OpenAI v1.x+
from openai.types.chat import (
    ChatCompletionMessageParam, 
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletion # To get CompletionUsage if needed for type hint
)
from openai.types.completion_usage import CompletionUsage # Explicit import for usage stats

from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv() # Load environment variables from .env file

# --- Pydantic Models for Request and Response ---

class Paper(BaseModel):
    """Defines the structure for a single paper."""
    id: str | int 
    title: str
    abstract: str

class SummarizationRequest(BaseModel):
    """Defines the structure for the summarization request body."""
    papers: List[Paper]
    topic_name: str = Field(default="Default Topic", description="A name for the collection of papers being summarized.")

class ReferenceItem(BaseModel):
    """Defines the structure for a reference item in the response."""
    id: str | int
    title: str

class SummarizationResponse(BaseModel):
    """Defines the structure for the summarization response."""
    topic_name: str
    summary: str
    references: List[ReferenceItem]
    tokens_used: Dict[str, int] | None = None 

# --- Global Variables & Initialization ---

app = FastAPI(
    title="Scientific Paper Summarization API",
    description="An API that receives a list of scientific papers and returns an AI-generated summary.",
    version="1.0.0"
)

PROMPTS_FILE = 'system_prompts.yaml' 
SYSTEM_PROMPTS: Dict[str, Any] = {}
try:
    with open(PROMPTS_FILE, 'r') as file:
        loaded_yaml = yaml.safe_load(file)
        if isinstance(loaded_yaml, dict) and 'prompts' in loaded_yaml:
            SYSTEM_PROMPTS = loaded_yaml['prompts']
        else:
            print(f"Warning: '{PROMPTS_FILE}' is not structured as expected or 'prompts' key is missing.")
            SYSTEM_PROMPTS = {"default_summary": {"content": "Provide a concise summary of the following scientific papers."}}
except FileNotFoundError:
    print(f"Warning: System prompts file '{PROMPTS_FILE}' not found. Using a default prompt.")
    SYSTEM_PROMPTS = {"default_summary": {"content": "Provide a concise summary of the following scientific papers."}}
except yaml.YAMLError as e:
    print(f"Error parsing YAML from '{PROMPTS_FILE}': {e}. Using a default prompt.")
    SYSTEM_PROMPTS = {"default_summary": {"content": "Provide a concise summary of the following scientific papers."}}

OPENAI_API_HOST = os.getenv("OPENAI_API_HOST")
OPENAI_API_PORT = os.getenv("OPENAI_API_PORT") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
MODEL_NAME = os.getenv("MODEL") # This can be str | None

# Runtime checks to ensure essential variables are set
if not OPENAI_API_HOST: 
    raise RuntimeError("OPENAI_API_HOST environment variable must be set.")
if not MODEL_NAME: 
    # This ensures MODEL_NAME is a str after this block, or the app doesn't start.
    raise RuntimeError("MODEL environment variable must be set.")


if "localhost" in OPENAI_API_HOST or "127.0.0.1" in OPENAI_API_HOST:
    base_url = f"{OPENAI_API_HOST}:{OPENAI_API_PORT}/v1/" if OPENAI_API_PORT else f"{OPENAI_API_HOST}/v1/"
else:
    base_url = f"{OPENAI_API_HOST}/" 

AI_CLIENT = Client(
    base_url=base_url,
    api_key=OPENAI_API_KEY if OPENAI_API_KEY else "not_needed" 
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

def generate_ai_response(client: Client, messages: List[ChatCompletionMessageParam], model: str) -> tuple[str, CompletionUsage | None]:
    """
    Generates a response from the AI model.
    Returns a tuple of (summary_text, usage_stats).
    Raises HTTPException if summary content is not generated.
    """
    try:
        response_object: ChatCompletion = client.chat.completions.create(
            model=model, 
            messages=messages, 
            stream=False,
            max_tokens=1000, 
            temperature=0.7, 
            top_p=0.95       
        )
        
        if not response_object.choices or not response_object.choices[0].message:
            print("Error: AI response missing choices or message.")
            raise HTTPException(status_code=500, detail="AI response malformed: missing choices or message.")

        summary_content = response_object.choices[0].message.content
        
        if summary_content is None or not summary_content.strip():
            print("Error: AI response content is None or empty.")
            raise HTTPException(status_code=500, detail="AI failed to generate summary content.")
            
        usage_stats = response_object.usage 
        return summary_content, usage_stats
    except HTTPException: 
        raise
    except Exception as e:
        print(f"Error during AI model call: {e}")
        raise HTTPException(status_code=503, detail=f"AI service unavailable or error: {str(e)}")

# --- API Endpoint ---

@app.post("/summarize/", response_model=SummarizationResponse)
async def summarize_papers_endpoint(request_data: SummarizationRequest = Body(...)):
    """
    Receives a list of papers and returns an AI-generated summary.
    """
    if not request_data.papers:
        raise HTTPException(status_code=400, detail="No papers provided for summarization.")

    selected_prompt_key = "single_paragraph" 
    
    system_prompt_content: str
    if selected_prompt_key in SYSTEM_PROMPTS and isinstance(SYSTEM_PROMPTS.get(selected_prompt_key), dict) and 'content' in SYSTEM_PROMPTS[selected_prompt_key]:
        system_prompt_content = SYSTEM_PROMPTS[selected_prompt_key]['content']
    elif SYSTEM_PROMPTS: 
        first_key = next(iter(SYSTEM_PROMPTS))
        system_prompt_content = SYSTEM_PROMPTS[first_key].get('content', "Summarize the provided texts.")
        print(f"Warning: Prompt key '{selected_prompt_key}' not found. Using fallback prompt '{first_key}'.")
    else: 
        system_prompt_content = "Provide a concise summary of the following scientific papers."
        print("Warning: No system prompts loaded. Using a very basic default prompt.")

    formatted_papers_string = format_papers_for_prompt(request_data.papers)
    ai_messages = create_ai_messages(system_prompt_content, formatted_papers_string)

    # Assert that MODEL_NAME is a string at this point due to startup checks.
    # This helps Pylance understand the type.
    assert isinstance(MODEL_NAME, str), "MODEL_NAME should be a string due to startup checks."
    summary_text, usage_stats = generate_ai_response(AI_CLIENT, ai_messages, MODEL_NAME)


    references_list = [ReferenceItem(id=p.id, title=p.title) for p in request_data.papers]
    
    token_info = None
    if usage_stats: 
        token_info = {
            "prompt_tokens": usage_stats.prompt_tokens,
            "completion_tokens": usage_stats.completion_tokens,
            "total_tokens": usage_stats.total_tokens,
        }

    return SummarizationResponse(
        topic_name=request_data.topic_name,
        summary=summary_text, 
        references=references_list,
        tokens_used=token_info
    )
