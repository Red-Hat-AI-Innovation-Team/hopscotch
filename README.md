# Hopscotch: A Method for Skipping Redundant Attention Blocks

This repository implements the Hopscotch method for skipping redundant transformer blocks while maintaining model performance through adaptive scaling parameters.

## Paper

**[Hopscotch: Discovering and Skipping Redundancies in Language Models](https://arxiv.org/abs/2506.03303)**

Hopscotch identifies and removes attention blocks that contribute least to task performance while maintaining output quality through adaptive scaling. The method jointly optimizes which blocks to skip and how to scale the outputs of the remaining layers, achieving minimal performance degradation (less than 2% drop when skipping 4 blocks on Llama-3.1-8B and Qwen2.5-7B).

## Requirements

### Special Transformers Fork

This implementation requires a special fork of transformers with scaling parameter support:

```bash
pip install git+https://github.com/Maxusmusti/transformers.git@training-scaling
```

### Other Dependencies

```bash
pip install -r requirements.txt
```

## Supported Models

### Tested Models

The following models have been tested with this implementation:

- **Llama 3.1 8B Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
- **Qwen 2.5 7B Instruct** (`Qwen/Qwen2.5-7B-Instruct`)
- **Granite 3.0 8B Instruct** (`ibm-granite/granite-3.0-8b-instruct`)

### Model Family Support

The transformers fork includes scaling parameter support for the following model families:

- **Llama models**: All Llama/Llama2/Llama3 variants (including Code Llama, Alpaca, Vicuna, etc.)
- **Qwen models**: Qwen 1.5, Qwen 2.0, Qwen 2.5 series
- **Granite models**: IBM Granite 3.0 series

Any model using these modeling classes should work with the scaling parameters, though they may require testing and validation for optimal performance.

### Adding Support for New Models

To add support for additional models, you need to implement scaling parameters in the decoder blocks:

1. **Add scaling parameters** to the decoder layer class:
```python
self.b_scale_attn = nn.Parameter(torch.ones(1))
self.s_scale_attn = nn.Parameter(torch.ones(1))
self.b_scale_mlp = nn.Parameter(torch.ones(1))
self.s_scale_mlp = nn.Parameter(torch.ones(1))
```

2. **Apply scaling in the forward function** after attention and MLP computations:
```python
# After attention
hidden_states *= self.b_scale_attn
hidden_states = residual * self.s_scale_attn + hidden_states

# After MLP
hidden_states *= self.b_scale_mlp
hidden_states = residual * self.s_scale_mlp + hidden_states
```

## Complete Workflow

### Step 1: Generate Ground Truth Data

Generate model responses for your dataset:

```bash
python generate_model_gt_data.py --model qwen
python generate_model_gt_data.py --model llama
python generate_model_gt_data.py --model granite
```

**Output**: `{model}_math_final.jsonl`

### Step 2: Process Data

Process the generated data using InstructLab's data processing:

```bash
python process_gt_data.py --model qwen
python process_gt_data.py --model llama --max_seq_len 25000
python process_gt_data.py --model granite --output_dir custom_output
```

**Output**: `data/{model}-processed-data/data.jsonl`

### Step 3: Block Selection (Iterative)

Identify the best blocks to remove through iterative training:

```bash
# First iteration - find first block to remove
python block_selection.py --model qwen --data_path data/qwen-processed-data/data.jsonl
# Output: "Best block to remove: 15 (loss: 2.1)"

# Second iteration - add previously selected block
python block_selection.py --model qwen --data_path data/qwen-processed-data/data.jsonl --starting_skipped_blocks 15
# Output: "Best block to remove: 8 (loss: 2.3)"

# Third iteration - continue building the list
python block_selection.py --model qwen --data_path data/qwen-processed-data/data.jsonl --starting_skipped_blocks 15 8
# Continue until desired number of blocks selected
```

**Resume capability**: If training is interrupted, resume with previous losses:
```bash
python block_selection.py --model qwen --data_path data/qwen-processed-data/data.jsonl --starting_skipped_blocks 15 8 --prev 2.1 2.3
```

### Step 4: Generate Final Checkpoints

#### Option A: Scaled Checkpoints (With Training)

Train scaling parameters for the selected blocks:

```bash
python param_scale.py --model qwen \
    --data_path data/qwen-processed-data/data.jsonl \
    --blocks_to_skip 15 8 \
    --num_epochs 10
```

**Output**: `scaled_qwen_ckpts/qwen_scaled_epoch_{0-9}/`

#### Option B: Unscaled Checkpoints (No Training)

Create a checkpoint with blocks set to zero (no scaling training):

```bash
python unscaled_ckpt.py --model qwen --blocks_to_skip 15 8
```

**Output**: `unscaled_qwen_ckpts/qwen_unscaled/`

## Script Parameters

### generate_model_gt_data.py
- `--model`: Model choice (qwen/llama/granite)

### process_gt_data.py
- `--model`: Model choice (qwen/llama/granite)
- `--input_file`: Custom input file (default: `{model}_math_final.jsonl`)
- `--output_dir`: Output directory (default: `data/{model}-processed-data`)
- `--max_seq_len`: Maximum sequence length (default: 29000)
- `--num_cpu_procs`: CPU processes for processing (default: 8)

### block_selection.py
- `--model`: Model choice (qwen/llama/granite)
- `--data_path`: Path to processed data (required)
- `--starting_skipped_blocks`: Previously selected blocks for removal
- `--prev`: Previous losses for resuming training

### param_scale.py
- `--model`: Model choice (qwen/llama/granite)
- `--data_path`: Path to processed data (required)
- `--blocks_to_skip`: Layer indices to skip/remove (required)
- `--num_epochs`: Training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 3e-3)
- `--batch_size`: Batch size (default: 32)

### unscaled_ckpt.py
- `--model`: Model choice (qwen/llama/granite)
- `--blocks_to_skip`: Layer indices to skip/remove (required)

## Output Structure

```
hopscotch/
├── {model}_math_final.jsonl              # Generated data
├── data/{model}-processed-data/data.jsonl # Processed data
├── scaled_{model}_ckpts/                  # Scaled checkpoints
│   ├── {model}_scaled_epoch_0/
│   ├── {model}_scaled_epoch_1/
│   └── ...
└── unscaled_{model}_ckpts/               # Unscaled checkpoints
    └── {model}_unscaled/
```

## Example: Complete Qwen Workflow

```bash
# 1. Generate data
python generate_model_gt_data.py --model qwen

# 2. Process data
python process_gt_data.py --model qwen

# 3. Find blocks to remove (repeat as needed)
python block_selection.py --model qwen --data_path data/qwen-processed-data/data.jsonl
python block_selection.py --model qwen --data_path data/qwen-processed-data/data.jsonl --starting_skipped_blocks 15
python block_selection.py --model qwen --data_path data/qwen-processed-data/data.jsonl --starting_skipped_blocks 15 8

# 4. Create final model
python param_scale.py --model qwen --data_path data/qwen-processed-data/data.jsonl --blocks_to_skip 15 8 --num_epochs 10
```

## Evaluation

### LM Eval Harness Compatibility

This implementation was evaluated using [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness). However, a small code change is required in the LM Eval Harness codebase to work with models that have skipped blocks:

**File**: `lm_eval/models/huggingface.py`

**Changes needed**:
1. **Line 1003** (`_model_generate`): Change `use_cache=True,` to `use_cache=False,`
2. **Line 967** (`_model_call`): Add `use_cache=False` parameter

**Example**:
```python
# Line 967 in _model_call
outputs = self._model(**inps, use_cache=False)

# Line 1003 in _model_generate
use_cache=False,
```

**Note**: This is an implementation issue, not a methodology issue. The transformers library returns missing values when `use_cache=True` for decoder layers with skipped blocks. This can be fixed with an update to the transformers fork in future versions.