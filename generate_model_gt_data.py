import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import vllm  # Import the vllm library

# Define model configurations
MODEL_CONFIGS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "granite": "ibm-granite/granite-3.0-8b-instruct"
}

def get_args():
    parser = argparse.ArgumentParser(description="Generate model ground truth data using different language models")
    parser.add_argument("--model", choices=["qwen", "llama", "granite"], required=True,
                       help="Choose the model to use: qwen, llama, or granite")
    return parser.parse_args()

# Parse command line arguments
args = get_args()
model_name = MODEL_CONFIGS[args.model]

print(f"Using model: {args.model} -> {model_name}")

# Define the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Initialize the vllm model (Use LLM from vllm)
model = vllm.LLM(model_name, dtype=torch.bfloat16, tensor_parallel_size=4, device="cuda")

# Ensure the tokenizer pads correctly
if tokenizer.unk_token:
    tokenizer.pad_token_id = tokenizer.unk_token_id
elif tokenizer.eos_token:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Loop through the input JSONL data and process it
new_jsonl = []
with open("gsm8k.jsonl", 'r') as math_data:
    for line in tqdm(math_data):
        line = line.strip()
        data = json.loads(line)
        data["messages"].pop(1)  # Remove the second message in the list
        
        # Use tokenize=False to avoid re-tokenization
        formatted_input = tokenizer.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=True)

        # Prepare the vllm inference request with the formatted tokens
        sampling_params = vllm.SamplingParams(
            max_tokens=4096,
            temperature=0,  # Greedy decoding (no sampling)
        )

        # Perform inference using vllm
        response = model.generate(formatted_input, sampling_params)

        # Extract the generated text from the response
        print(response)
        generated_text = response[0].outputs[0].text.strip()  # vllm returns a list of responses
        

        print("-----------------------------------------")
        print(f"Generated text: {generated_text}")
        print("-----------------------------------------")

        # Append the generated text to the data structure
        data["messages"].append({"role": "assistant", "content": generated_text})
        with open(f"{args.model}_math.jsonl",'a') as f:
            f.write(json.dumps(data) + '\n')
        new_jsonl.append(data)

# Write all formatted data to JSONL
with open(f"{args.model}_math_final.jsonl", 'w') as f:
    f.write('\n'.join(map(json.dumps, new_jsonl)))

