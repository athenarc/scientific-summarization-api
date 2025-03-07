from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pprint import pprint
import textwrap
import yaml
import requests
from dotenv import load_dotenv
import os
from openai import Client


def load_prompts(yaml_file='system_prompts.yaml'):
    """Load system prompts from a YAML file."""
    with open(yaml_file, 'r') as file:
        prompts = yaml.safe_load(file)
    return prompts


def read_data_file(data_file):
    """Open a JSON data file and return its contents as a dictionary."""

    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def initialize_client(host, port, api_key):
    """Initialize the OpenAI client."""

    client = Client(
        base_url=f"{host}:{port}/v1/",
        api_key=api_key
    )
    
    return client


def bip_api_request(bip_url, keywords, auth_token, page_size=5):
    """Make a request to the BIP API to search for papers based on keywords."""

    url = f"{bip_url}/paper/search?keywords={keywords}&page=1&page_size={page_size}&auth_token={auth_token}"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    
    response_json = response.json()

    for i, row in enumerate(response_json['rows'], start=1):
        row['id'] = i

    return response_json


def format_papers_for_topic(papers):
    """Format papers for a specific topic into a structured string."""

    formatted_output = "\nPapers to analyze:\n\n"

    papers = sorted(papers, key=lambda x: x['id'])
    
    for paper in papers:
        formatted_output += f"Paper ID: {paper['id']}\n"
        formatted_output += f"Title: {paper['title']}\n"
        formatted_output += f"Abstract: {paper['abstract']}\n"
        formatted_output += "-" * 80 + "\n\n"
    
    return formatted_output


def create_conversation_messages(system_prompt, data):
    """Create conversation messages for each topic in the JSON file."""

    topic_messages = {}
    
    for topic, papers in data.items():
        formatted_papers = format_papers_for_topic(papers)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_papers}
        ]

        topic_messages[topic] = messages

    return topic_messages


def get_topic_stats(topic, papers):
    """Print the number of papers for a given topic."""

    num_papers = len(papers)
    print(f"Processing topic: {topic}")
    print(f"Number of papers in topic: {num_papers}")


def get_number_of_tokens(response, usage):
    """Calculate the number of tokens for the system prompt, papers, and text."""

    prompt_tokens = usage.prompt_tokens
    response_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    total_words = len(response.split())
    total_tokens = prompt_tokens + response_tokens
    
    print(f"Number of tokens in prompt: {prompt_tokens}")
    print(f"Number of tokens in output: {response_tokens}")
    print(f"Total number of words in output: {total_words}")
    print(f"Total number of tokens: {total_tokens}")


def generate_response(client, messages, model="tgi"):
    """Generate a response from the model based on the provided messages."""

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=1000,
        temperature=0.7,
        top_p=0.95
    )

    usage = response.usage
    response = response.choices[0].message.content

    return response, usage


def pretty_print_model_response(topic, papers, response, width=100, new_response=False):
    """Pretty print the model's response with academic formatting."""

    print("\n" + "="*width)
    if new_response:
        print("UPDATED SUMMARY FOR TOPIC: " + topic.upper())
    else:
        print(f"TOPIC: {topic.upper()}")
    print("="*width + "\n")

    sections = response.split('\n\n')
    
    for section in sections:
        formatted_section = textwrap.fill(
            section.strip(),
            width=width,
            initial_indent="",
            subsequent_indent="",
            # subsequent_indent="    "
        )
        print(formatted_section + "\n")
    
    print("References:")
    for paper in papers:
        print(f"[{paper['id']}] {paper['title']}")
    
    print("="*width + "\n")


def rewrite_request(client, system_prompt, summary, word_limit=250, model="tgi"):
    """Generate a shorter summary if the word limit is exceeded."""

    rewrite_request = f"""
    The summary provided exceeds the word limit of {word_limit} words. Please revise the summary to be shorter and more concise and adhere to the {word_limit}-word maximum limit.
    
    {system_prompt}
    ------
    Summary:
    {summary}
    """

    messages = [
        {"role": "user", "content": rewrite_request}
    ]

    response, usage = generate_response(
        client=client,
        messages=messages, 
        model=model
    )
    
    return response, usage


def get_summary(system_prompt, client, data_input, request_origin="api", model="tgi", response_only=False, print_response=True, show_papers=False, word_limit=250):
    """Generate a summary based on the provided system prompt, model, and tokenizer."""

    # Load data based on origin
    if request_origin == 'api':
        data = {
            data_input: bip_api_request(
                bip_url=os.getenv("BIP_URL"), 
                keywords=data_input,
                auth_token=os.getenv("BIP_AUTH_TOKEN"))['rows']
        }
    elif request_origin == 'file':
        data = read_data_file(data_input)
    
    # Create conversation messages for each topic
    topic_messages = create_conversation_messages(
        system_prompt=system_prompt,
        data=data
    )
    
    # Store results for all topics
    results = {}
    
    for topic, messages in topic_messages.items():
        papers = data[topic]
        
        # Generate initial response
        response, usage = generate_response(
            client=client,
            messages=messages,
            model=model)
        total_words = len(response.split())

        # Display information based on parameters
        if print_response:
            # Show topic statistics if not response_only
            if not response_only:
                get_topic_stats(
                    topic=topic,
                    papers=papers
                )
                if show_papers:
                    print(format_papers_for_topic(papers))
            
            # Print the initial response
            pretty_print_model_response(
                topic=topic,
                papers=papers, 
                response=response
            )

            if not response_only:
                get_number_of_tokens(
                    response=response,
                    usage=usage
                )
            
            # If verbose and we had to rewrite, show this was an updated response
            if total_words > word_limit:
                response, usage = rewrite_request(
                    client=client,
                    system_prompt=system_prompt,
                    summary=response,
                    word_limit=word_limit,
                    model=model
                )
                
                if print_response:
                    print(f"\nThe summary exceeds the word limit of {word_limit} words.")
                    pretty_print_model_response(
                        topic=topic,
                        papers=papers,
                        response=response, 
                        new_response=True
                    )
                    
                    # Show token counts for rewritten response if not in response_only mode
                    if not response_only:
                        get_number_of_tokens(
                            response=response,
                            usage=usage
                        )

        # Store the result
        results[topic] = response
    
    # Return all results or just the last one for backwards compatibility
    if len(results) == 1:
        return next(iter(results.values()))
    return results