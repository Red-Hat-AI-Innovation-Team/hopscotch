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
    parser = argparse.ArgumentParser(description="Parameter scaling training for block removal")
    parser.add_argument("--model", choices=["qwen", "llama", "granite"], required=True,
                       help="Choose the model to use: qwen, llama, or granite")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the processed data JSONL file")
    parser.add_argument("--blocks_to_skip", nargs='*', type=int, required=True,
                       help="Layer indices to skip/remove (from block selection results)")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=3e-3,
                       help="Learning rate (default: 3e-3)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    return parser.parse_args()

# Parse command line arguments
args = get_args()
model_name = MODEL_CONFIGS[args.model]

print(f"Using model: {args.model} -> {model_name}")
print(f"Data path: {args.data_path}")
print(f"Blocks to skip: {args.blocks_to_skip}")
print(f"Number of epochs: {args.num_epochs}")
print(f"Learning rate: {args.learning_rate}")
print(f"Batch size: {args.batch_size}")

# Create output directory
output_dir = f"scaled_{args.model}_ckpts"
os.makedirs(output_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {output_dir}")

################HYPERPARAMS#####################
num_epochs = args.num_epochs
to_freeze = args.blocks_to_skip  # Blocks to skip/remove from block selection results
learning_rate = args.learning_rate
lambda_l1 = 0 #1e-3  # Regularization strength
batch_size = args.batch_size
data_path = args.data_path
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


#print(to_optimize)
optimizer = torch.optim.Adam(
    to_optimize,  # Only optimize scaling factors
    lr=learning_rate
)
loss_fn = torch.nn.CrossEntropyLoss()

dataset = setup_dataset(data_path)
dataloader = setup_dataloader(dataset, tokenizer.pad_token_id,max_batch_len=60000,flash_enabled=True,samples_per_gpu=batch_size,sampler="distributed")
for epoch in range(num_epochs):
    #dataloader.sampler.set_epoch(epoch)
    print(f"STARTING EPOCH {epoch}")

    losses = []
    for batch in tqdm(dataloader, desc="Training Progress"):
        #print(len(batch))
        #print(batch)
        # Perform inference
        num_loss_counted_tokens = float(
            torch.tensor([batch.pop("num_loss_counted_tokens")])
        )
        micro_batch_size = float(torch.tensor([batch.pop("num_samples")]))
        for k in batch:
            batch[k] = batch[k].to('cuda')

        print(f"Epoch {epoch} - num samples in batch: {micro_batch_size}")

        optimizer.zero_grad()

        outputs = model(**batch, use_cache=False)
        loss = outputs.loss

        # Compute L1 regularization
        l1_norm = sum(p.abs().sum().to('cuda:0') for p in to_optimize)
        print("L1 Norm: ", l1_norm.item())
        print("lambda * L1: ", (lambda_l1 * l1_norm).item())

        # Add L1 regularization to the loss
        loss = loss + lambda_l1 * l1_norm
        del l1_norm

        print("Loss", loss.item())
        print('\n')

        losses.append(loss)
        # Backward pass to compute gradients
        loss.backward()

        # Update parameters (only scaling factors will be updated)
        optimizer.step()

    print("avg loss: ", sum(losses)/len(losses))
    print("\nCURRENT STATE OF SCALING PARAMETERS:")
    for name, param in model.named_parameters():
        # If the parameter is not in the pre-trained checkpoint (uninitialized)
        if "scale" in name:
            print(name, param.item())

    checkpoint_name = f"{args.model}_scaled_epoch_{epoch}"
    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    print(f"\nSAVING MODEL CHECKPOINT TO {checkpoint_path}\n")

    # Save the trained model
    model.save_pretrained(checkpoint_path)

    # Save the tokenizer (important for future tokenization)
    tokenizer.save_pretrained(checkpoint_path)

    #hidden_states = outputs.hidden_states  # tuple of hidden states
    #num_layers = len(hidden_states) - 1   # excluding the embeddings

    # Iterate through the hidden states for each layer
"""
    for layer_idx, layer_hidden in enumerate(hidden_states[1:]):  # Skip the embedding layer
        print(f"Layer {layer_idx + 1} hidden state shape: {layer_hidden.shape}")

        # layer_hidden has shape (batch_size, seq_len, hidden_size)
        # You can further process it here, or store it as needed
        # Example: if you want the last token's hidden state for each layer:
        last_token_hidden = layer_hidden[:, -1, :]  # Extract the hidden state of the last token
        print(f"Last token hidden state at layer {layer_idx + 1}: {last_token_hidden}")
"""

#[1,1,1,1,1,1,1,1,1,1,1,1,1,1.5,0,1,1.5,0,1,1,1.5,0,1,1.5,0,1,1,1,1,1,1,1]

