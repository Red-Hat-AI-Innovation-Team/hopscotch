import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from instructlab.training.token_dataset import setup_dataloader, setup_dataset
from tqdm import tqdm

# Define model configurations
MODEL_CONFIGS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "granite": "ibm-granite/granite-3.0-8b-instruct"
}

def get_args():
    parser = argparse.ArgumentParser(description="Create unscaled checkpoint with specified blocks skipped")
    parser.add_argument("--model", choices=["qwen", "llama", "granite"], required=True,
                       help="Choose the model to use: qwen, llama, or granite")
    parser.add_argument("--blocks_to_skip", nargs='*', type=int, required=True,
                       help="Layer indices to skip/remove (from block selection results)")
    return parser.parse_args()

# Parse command line arguments
args = get_args()
model_name = MODEL_CONFIGS[args.model]

print(f"Using model: {args.model} -> {model_name}")
print(f"Blocks to skip: {args.blocks_to_skip}")

# Create output directory
output_dir = f"unscaled_{args.model}_ckpts"
os.makedirs(output_dir, exist_ok=True)
checkpoint_name = f"{args.model}_unscaled"
checkpoint_path = os.path.join(output_dir, checkpoint_name)
print(f"Checkpoint will be saved to: {checkpoint_path}")

################HYPERPARAMS#####################
to_freeze = args.blocks_to_skip  # Blocks to skip/remove from block selection results
################################################

# Define the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map='auto')
model.train()

to_optimize = []
for name, param in model.named_parameters():
    # If the parameter is not in the pre-trained checkpoint (uninitialized)
    if "scale" in name:
        ind = int(name.split('.')[2])
        if ind in to_freeze and "attn" in name:
            torch.nn.init.constant_(param, 0)
            param.requires_grad = False
            print(f"Froze {name} at 0")
        elif param.requires_grad:  # Only initialize trainable parameters
            # Initialize to 1
            torch.nn.init.constant_(param, 1)
            to_optimize.append(param)
            print(f"Initialized {name} to 1")
    else:
        param.requires_grad = False

print(f"\nSAVING UNSCALED MODEL CHECKPOINT TO {checkpoint_path}\n")

# Save the trained model
model.save_pretrained(checkpoint_path)

# Save the tokenizer (important for future tokenization)
tokenizer.save_pretrained(checkpoint_path)

print(f"Unscaled checkpoint saved successfully!")
print(f"Model saved to: {checkpoint_path}")

exit()
