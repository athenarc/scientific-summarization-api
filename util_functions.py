import argparse
import json
import textwrap
import yaml
import requests
from dotenv import load_dotenv
import os
from openai import Client

def load_prompts(yaml_file='system_prompts.yaml'):
    """Load system prompts from a YAML file."""
    # Open the YAML file in read mode
    with open(yaml_file, 'r') as file:
        # Load the YAML content safely
        prompts = yaml.safe_load(file)
    return prompts


def read_data_file(data_file):
    """Open a JSON data file and return its contents as a dictionary."""
    # Open the JSON file in read mode with UTF-8 encoding
    with open(data_file, 'r', encoding='utf-8') as file:
        # Load the JSON data
        data = json.load(file)
    return data


def initialize_client(host, port, api_key):
    """Initialize the OpenAI client."""
    # Check if the host is localhost to construct the base URL accordingly
    # Ensure OPENAI_API_HOST is fetched correctly, providing a default empty string if not set
    api_host_env = os.getenv("OPENAI_API_HOST", "")
    if "localhost" in api_host_env: # Compare with the fetched environment variable
        base_url=f"{host}:{port}/v1/"
    else:
        base_url=f"{host}/"

    # Create an OpenAI Client instance
    client = Client(
        base_url=base_url,
        api_key=api_key
    )
    
    return client


def bip_api_request(bip_url, keywords, auth_token, page_size=5): # page_size already defaults to 5
    """Make a request to the BIP API to search for papers based on keywords."""
    # Construct the API request URL, using the page_size parameter
    url = f"{bip_url}/paper/search?keywords={keywords}&page=1&page_size={page_size}&auth_token={auth_token}"

    payload = {}
    headers = {}

    # Make a GET request to the API
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during BIP API request: {e}")
        return {"rows": []} # Return an empty structure on error
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from BIP API response: {e}")
        return {"rows": []} # Return an empty structure on error


    # Add a sequential 'id' to each paper row for easier reference, only if 'rows' exists and is a list
    if 'rows' in response_json and isinstance(response_json['rows'], list):
        for i, row in enumerate(response_json['rows'], start=1):
            row['id'] = i
    else:
        print("Warning: 'rows' key missing or not a list in BIP API response.")
        response_json['rows'] = [] # Ensure 'rows' exists even if API returned unexpected structure

    return response_json


def format_papers_for_topic(papers):
    """Format papers for a specific topic into a structured string."""
    # Initialize the output string with a header
    formatted_output = "\nPapers to analyze:\n\n"

    # Sort papers by their 'id'
    # Ensure papers is a list and items have 'id' before sorting
    if isinstance(papers, list) and all(isinstance(p, dict) and 'id' in p for p in papers):
        papers = sorted(papers, key=lambda x: x['id'])
    else:
        print("Warning: Papers data is not in the expected format for sorting.")
        return "Error: Could not format papers due to data issue.\n"

    # Iterate through each paper and append its details to the output string
    for paper in papers:
        formatted_output += f"Paper ID: {paper.get('id', 'N/A')}\n"
        formatted_output += f"Title: {paper.get('title', 'N/A')}\n"
        formatted_output += f"Abstract: {paper.get('abstract', 'N/A')}\n"
        formatted_output += "-" * 80 + "\n\n"  # Separator line
    
    return formatted_output


def create_conversation_messages(system_prompt, data):
    """Create conversation messages for each topic in the JSON file."""
    # Initialize a dictionary to store messages for each topic
    topic_messages = {}
    
    # Iterate through each topic and its associated papers in the input data
    for topic, papers in data.items():
        # Format the papers for the current topic
        formatted_papers = format_papers_for_topic(papers)

        # Create the message structure for the AI model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_papers}
        ]

        # Store the messages for the current topic
        topic_messages[topic] = messages

    return topic_messages


def get_topic_stats(topic, papers):
    """Print the number of papers for a given topic."""
    # Get the number of papers for the topic
    num_papers = len(papers) if isinstance(papers, list) else 0
    print(f"Processing topic: {topic}")
    print(f"Number of papers in topic: {num_papers}")


def get_number_of_tokens(response_text, usage): # Renamed response to response_text to avoid conflict
    """Calculate and print the number of tokens for the prompt, output, and total."""
    # Extract token counts from the usage object
    prompt_tokens = usage.prompt_tokens
    response_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens 
    
    # Calculate total words in the response
    total_words = len(response_text.split())
    
    print(f"Number of tokens in prompt: {prompt_tokens}")
    print(f"Number of tokens in output: {response_tokens}")
    print(f"Total number of words in output: {total_words}")
    print(f"Total number of tokens (from API): {total_tokens}")


def generate_response(client, messages, model="tgi"):
    """Generate a response from the model based on the provided messages."""
    try:
        # Create a chat completion request to the AI model
        response_object = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,        # Do not stream the response
            max_tokens=1000,     # Maximum tokens for the generated response
            temperature=0.7,     # Controls randomness: lower is more deterministic
            top_p=0.95           # Nucleus sampling: considers tokens with top_p probability mass
        )

        # Extract usage statistics and the content of the response
        usage = response_object.usage
        response_content = response_object.choices[0].message.content

        return response_content, usage
    except Exception as e: # Catch potential errors from the API call
        print(f"Error generating response from AI model: {e}")
        return "Error: Could not generate summary.", None # Return a placeholder and None for usage


def pretty_print_model_response(topic, papers, response, width=100):
    """Pretty print the model's response with academic formatting."""
    # Print a header for the topic
    print("\n" + "="*width)
    print(f"TOPIC: {topic.upper()}")
    print("="*width + "\n")

    # Split the response into sections (paragraphs)
    sections = response.split('\n\n')
    
    # Format and print each section with text wrapping
    for section in sections:
        formatted_section = textwrap.fill(
            section.strip(),
            width=width,
            initial_indent="",
            subsequent_indent="", 
        )
        print(formatted_section + "\n")
    
    # Print the list of references
    print("References:")
    if isinstance(papers, list): # Ensure papers is a list
        for paper in papers: 
            print(f"[{paper.get('id','N/A')}] {paper.get('title', 'N/A')}")
    
    print("="*width + "\n")


def get_summary(system_prompt, client, data_input, request_origin="api", model="tgi", 
                response_only=False, print_response=True, show_papers=False, 
                num_papers_api=5): # Added num_papers_api with default 5
    """Generate a summary based on the provided system prompt, model, and input data."""

    # Load data based on the specified origin (API or file)
    if request_origin == 'api':
        # Fetch data from the BIP API, passing num_papers_api as page_size
        api_response = bip_api_request(
                bip_url=os.getenv("BIP_URL"), 
                keywords=data_input,
                auth_token=os.getenv("BIP_AUTH_TOKEN"),
                page_size=num_papers_api) # Pass num_papers_api here
        
        if 'rows' not in api_response or not isinstance(api_response['rows'], list): # Check if 'rows' is a list
            print(f"Error: 'rows' key not found or not a list in API response for keywords: {data_input}")
            print(f"API Response: {api_response}")
            return None
        data = { data_input: api_response['rows'] }

    elif request_origin == 'file':
        try:
            # Read data from a local JSON file
            data = read_data_file(data_input)
        except FileNotFoundError:
            print(f"Error: Data file '{data_input}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from data file '{data_input}'.")
            return None
    else:
        # Handle invalid request origin
        print(f"Error: Invalid request_origin '{request_origin}'. Must be 'api' or 'file'.")
        return None 
    
    # Create conversation messages for each topic based on the loaded data
    topic_messages = create_conversation_messages(
        system_prompt=system_prompt,
        data=data
    )
    
    # Initialize a dictionary to store results for all topics
    results = {}
    
    # Process each topic
    for topic, messages in topic_messages.items():
        current_papers = data.get(topic) # Get the papers for the current topic safely
        if not isinstance(current_papers, list):
            print(f"Warning: Papers for topic '{topic}' are not in the expected list format. Skipping.")
            results[topic] = "Error: Could not process papers for this topic."
            continue

        # Generate the AI response
        response_text, usage = generate_response( # Renamed response to response_text
            client=client,
            messages=messages,
            model=model)
        
        if usage is None: # Indicates an error in generate_response
            results[topic] = response_text # Store the error message
            continue


        # Display information based on parameters
        if print_response:
            # Show topic statistics if not in response_only mode
            if not response_only:
                get_topic_stats(
                    topic=topic,
                    papers=current_papers
                )
                # Show paper details if requested
                if show_papers:
                    print(format_papers_for_topic(current_papers))
            
            # Print the AI-generated response
            pretty_print_model_response(
                topic=topic,
                papers=current_papers, 
                response=response_text 
            )

            # Show token counts if not in response_only mode
            if not response_only:
                get_number_of_tokens(
                    response_text=response_text, # Pass response_text
                    usage=usage
                )
            
        # Store the final response for the topic
        results[topic] = response_text
    
    # Return all results, or just the single result if only one topic was processed
    if len(results) == 1:
        return next(iter(results.values()))
    return results
