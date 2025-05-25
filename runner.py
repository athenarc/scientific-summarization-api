from util_functions import *

def main():
    # Load environment variables from a .env file
    load_dotenv()

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Scientific Summarization Tool")
    # Define command-line arguments
    parser.add_argument('--request-origin', type=str, required=True, choices=['api', 'file'], help='Origin of the request (api or file)')
    parser.add_argument('--input', type=str, required=True, help='Input data (keywords for API, or filename for file)')
    parser.add_argument('--response-only', action='store_true', help='Show only the response without statistics')
    parser.add_argument('--show-papers', action='store_true', help='Show papers related to the input')
    # Added --num-papers argument
    parser.add_argument('--num-papers', type=int, default=5, help='Number of papers to fetch from API (default: 5)')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load system prompts from the YAML file
    try:
        prompts = load_prompts() 
    except FileNotFoundError:
        print("Error: 'system_prompts.yaml' not found. Please ensure the file exists in the correct location.")
        print("Using a default generic prompt instead.")
        prompts = {'prompts': {'single_paragraph': {'content': "Summarize the following scientific papers."}}}
    except yaml.YAMLError:
        print("Error: Could not parse 'system_prompts.yaml'. Please check its formatting.")
        print("Using a default generic prompt instead.")
        prompts = {'prompts': {'single_paragraph': {'content': "Summarize the following scientific papers."}}}

    # Initialize the OpenAI client using environment variables
    openai_api_host = os.getenv("OPENAI_API_HOST")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL") # Get model from environment variable

    if not openai_api_host: #Removed check for openai_api_key as it might not be needed for local models
        print("Error: OPENAI_API_HOST environment variable not set.")
        return
    if not model_name:
        print("Error: MODEL environment variable not set.")
        return
    # Note: openai_api_key might be an empty string for some local/self-hosted models.
    # The openai library typically handles this, but strict checks might prevent usage if key is optional.


    client = initialize_client(
        host=openai_api_host,
        port=os.getenv("OPENAI_API_PORT"), 
        api_key=openai_api_key if openai_api_key is not None else "not_needed"
    )

    # Select the appropriate system prompt
    system_prompt_key = 'single_paragraph' 
    selected_system_prompt = "Summarize the following scientific papers." # Default prompt
    if 'prompts' in prompts and isinstance(prompts.get('prompts'), dict) and \
       system_prompt_key in prompts['prompts'] and \
       isinstance(prompts['prompts'].get(system_prompt_key), dict) and \
       'content' in prompts['prompts'][system_prompt_key]:
        selected_system_prompt = prompts['prompts'][system_prompt_key]['content']
    else:
        print(f"Warning: System prompt for '{system_prompt_key}' not found or is malformed in system_prompts.yaml.")
        print("Using a default generic prompt.")


    # Call the main summarization function with parsed arguments and loaded configurations
    summary = get_summary(
        system_prompt=selected_system_prompt,
        client=client,
        request_origin=args.request_origin,
        model=model_name, 
        data_input=args.input,
        response_only=args.response_only,
        show_papers=args.show_papers,
        num_papers_api=args.num_papers
    )

    if summary and not args.response_only:
        print("\nSummarization process complete.")
    elif summary and args.response_only:
        # Output is handled within get_summary by pretty_print_model_response
        pass
    elif not summary:
        print("\nSummarization process could not be completed or returned no summary.")


if __name__ == "__main__":
    main()
