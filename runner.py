from util_functions import *
import argparse
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Scientific Summarization Tool")
    parser.add_argument('--request-origin', type=str, required=True, help='Origin of the request (api or file)')
    parser.add_argument('--input', type=str, required=True, help='Input data (keywords or filename)')
    parser.add_argument('--response-only', action='store_true', help='Show only the response without statistics')
    parser.add_argument('--show-papers', action='store_true', help='Show papers related to the input')
    # add optional argument for word limit and default to 250
    parser.add_argument('--word-limit', type=int, default=250, help='Word limit for the summary')
    args = parser.parse_args()

    prompts = load_prompts()
    client = initialize_client(
        host=os.getenv("OPENAI_API_HOST"),
        port=os.getenv("OPENAI_API_PORT"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    summary = get_summary(
        system_prompt=prompts['prompts']['single_paragraph']['content'],
        client=client,
        request_origin=args.request_origin,
        model=os.getenv("MODEL"),
        data_input=args.input,
        response_only=args.response_only,
        show_papers=args.show_papers,
        word_limit=args.word_limit
    )


if __name__ == "__main__":
    main()