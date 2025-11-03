import os
import argparse
from instructlab.training.data_process import process_messages_into_input_ids

# Define model configurations (same as generate_model_gt_data.py)
MODEL_CONFIGS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "granite": "ibm-granite/granite-3.0-8b-instruct"
}

def get_args():
    parser = argparse.ArgumentParser(description="Process ground truth data using instructlab data processing")
    parser.add_argument("--model", choices=["qwen", "llama", "granite"], required=True,
                       help="Choose the model to use: qwen, llama, or granite")
    parser.add_argument("--input_file", type=str,
                       help="Input JSONL file (if not provided, uses {model}_math_final.jsonl)")
    parser.add_argument("--output_dir", type=str, default="data/{model}-processed-data",
                       help="Output directory for processed data (default: data/{model}-processed-data)")
    parser.add_argument("--max_seq_len", type=int, default=29000,
                       help="Maximum sequence length (default: 29000)")
    parser.add_argument("--num_cpu_procs", type=int, default=8,
                       help="Number of CPU processes for data processing (default: 8)")
    return parser.parse_args()

def main():
    args = get_args()

    # Get model configuration
    model_name = MODEL_CONFIGS[args.model]

    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = f"{args.model}_math_final.jsonl"

    # Determine output directory
    if args.output_dir == "data/{model}-processed-data":
        output_dir = f"data/{args.model}-processed-data"
    else:
        output_dir = args.output_dir

    print(f"Processing ground truth data for model: {args.model}")
    print(f"Model path: {model_name}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"CPU processes: {args.num_cpu_procs}")

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process the data using instructlab's data processing
    print("\nStarting data processing...")

    processed_data_path = process_messages_into_input_ids(
        data_path=input_file,
        data_output_path=output_dir,
        model_path=model_name,
        max_seq_len=args.max_seq_len,
        num_cpu_procs=args.num_cpu_procs
    )

    print(f"\nData processing completed!")
    print(f"Processed data saved to: {processed_data_path}")
    print(f"You can now use '{output_dir}/data.jsonl' for training.")

if __name__ == "__main__":
    main()