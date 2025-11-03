import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from instructlab.training.token_dataset import setup_dataloader, setup_dataset
from tqdm import tqdm
from torch import tensor

# Define model configurations
MODEL_CONFIGS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "granite": "ibm-granite/granite-3.0-8b-instruct"
}

def get_args():
    parser = argparse.ArgumentParser(description="Block selection training with different language models")
    parser.add_argument("--model", choices=["qwen", "llama", "granite"], required=True,
                       help="Choose the model to use: qwen, llama, or granite")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the processed data JSONL file")
    parser.add_argument("--prev", nargs='*', type=float, default=[],
                       help="Previous average losses from already processed layers (for resuming)")
    parser.add_argument("--starting_skipped_blocks", nargs='*', type=int, default=[],
                       help="Layer indices that are already selected for removal from previous iterations")
    return parser.parse_args()

# Parse command line arguments
args = get_args()
model_name = MODEL_CONFIGS[args.model]

print(f"Using model: {args.model} -> {model_name}")
print(f"Data path: {args.data_path}")
if args.starting_skipped_blocks:
    print(f"Starting with skipped blocks: {args.starting_skipped_blocks}")
else:
    print("Starting with no skipped blocks")

if args.prev:
    print(f"Resuming from layer {len(args.prev)} with previous losses: {args.prev}")
else:
    print("Starting fresh block selection process")

################HYPERPARAMS#####################
num_epochs = 1
main_to_freeze = args.starting_skipped_blocks  # Blocks already selected for removal in previous iterations
learning_rate = 1e-2 #2e-3
lambda_l1 = 0 #1e-3  # Regularization strength
batch_size = 32
data_path = args.data_path  # Use the provided data path
################################################

# Get the number of hidden layers from model config
config = AutoConfig.from_pretrained(model_name)
num_hidden_layers = config.num_hidden_layers

print(f"Model has {num_hidden_layers} hidden layers")

prev = args.prev

average_losses = []
for i in range(num_hidden_layers):
    if i in range(len(prev)):
        average_losses.append(prev[i])
        continue
    if i in main_to_freeze:
        average_losses.append(100)
        continue
    to_freeze = main_to_freeze + [i]
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
        average_losses.append((sum(losses)/len(losses)).item())
        print("ALL AVERAGE LOSSES")
        print(average_losses)

min_ind = None
min_loss = 100
for ind, loss in enumerate(average_losses):
    if loss < min_loss:
        min_loss = loss
        min_ind = ind
        #print(ind)
print("\n\nMIN LOSS")
print("IND: ",min_ind)
print("LOSS: ", min_loss)

