from util_functions import *
import argparse
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Scientific Summarization Tool")
    parser.add_argument('--request_origin', type=str, required=True, help='Origin of the request (api or file)')
    parser.add_argument('--input', type=str, required=True, help='Input data (keywords or filename)')
    parser.add_argument('--response_only', action='store_true', help='Show only the response without statistics')
    parser.add_argument('--show_papers', action='store_true', help='Show papers related to the input')
    args = parser.parse_args()

    prompts = load_prompts()
    model_name = "Qwen/Qwen2.5-14B-Instruct-1M"
    model, tokenizer = initialize_model(model_name)

    summary = get_summary(
        system_prompt=prompts['prompts']['single_paragraph']['content'],
        model=model,
        tokenizer=tokenizer,
        request_origin=args.request_origin,
        data_input=args.input,
        response_only=args.response_only,
        show_papers=args.show_papers,
        word_limit=250
    )


if __name__ == "__main__":
    main()