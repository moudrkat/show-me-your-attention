# Fine-tuning TinyStories for Star Wars Stories

This guide shows you how to fine-tune the TinyStories model to generate Star Wars-themed children's stories.

## Overview

The fine-tuning pipeline consists of three main components:

1. **`config/training_config.py`** - Training hyperparameters and configuration
2. **`prepare_starwars_dataset.py`** - Dataset preparation utilities
3. **`finetune_starwars.py`** - Main training script

## Quick Start

### 1. Install Dependencies

First, install the required packages:

```bash
pip install torch transformers datasets accelerate
```

For LoRA (parameter-efficient fine-tuning):

```bash
pip install peft
```

### 2. Prepare Your Dataset

The dataset should contain Star Wars stories. You have several options:

#### Option A: Use Example Dataset (for testing)

Run the dataset preparation script to create a small example dataset:

```bash
python prepare_starwars_dataset.py
```

This creates `data/starwars_stories.json` with 10 example stories.

#### Option B: Use Your Own Stories

Create a text file with your Star Wars stories, separated by double newlines:

```text
Once upon a time, in a galaxy far away, Luke Skywalker...

On the planet Tatooine, a young boy named Anakin...

Princess Leia sent a secret message to...
```

Then modify `prepare_starwars_dataset.py` to load your file:

```python
# In the main() function, uncomment and modify:
stories = preparer.load_from_text_file("path/to/your/stories.txt")
```

#### Option C: Use JSONL Format

If you have stories in JSONL format:

```jsonl
{"text": "Once upon a time, in a galaxy far away..."}
{"text": "Luke Skywalker trained with Master Yoda..."}
```

Modify the script:

```python
stories = preparer.load_from_jsonl("path/to/your/stories.jsonl")
```

### 3. Configure Training

Edit `config/training_config.py` to adjust hyperparameters:

```python
@dataclass
class TrainingConfig:
    # Choose your model size
    model_name: str = "roneneldan/TinyStories-33M"  # 33M params (recommended)
    # model_name: str = "roneneldan/TinyStories-8M"  # 8M params (faster)

    # Training settings
    num_epochs: int = 3           # Number of training epochs
    batch_size: int = 4           # Batch size per GPU
    learning_rate: float = 5e-5   # Learning rate

    # LoRA (parameter-efficient training)
    use_lora: bool = False        # Set to True for LoRA
```

### 4. Run Fine-tuning

Basic training:

```bash
python finetune_starwars.py
```

With custom options:

```bash
# Use the smaller model
python finetune_starwars.py --model roneneldan/TinyStories-8M

# Use LoRA for efficient training
python finetune_starwars.py --use-lora

# Custom dataset and output directory
python finetune_starwars.py \
    --dataset data/my_starwars_stories.json \
    --output-dir models/my-starwars-model

# Adjust training parameters
python finetune_starwars.py \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 3e-5
```

### 5. Generate Stories

After training, the script automatically generates sample stories. To generate more:

```bash
# Generate from the trained model
python finetune_starwars.py --generate-only
```

Or use it in Python:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("models/tinystories-starwars/final")
tokenizer = AutoTokenizer.from_pretrained("models/tinystories-starwars/final")

prompt = "Once upon a time, in a galaxy far, far away, a young Jedi"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.8)
story = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(story)
```

## Advanced Usage

### LoRA Fine-tuning

LoRA (Low-Rank Adaptation) allows you to fine-tune large models efficiently by only training a small number of additional parameters.

**Benefits:**
- Much faster training
- Lower memory usage
- Smaller checkpoint files
- Can merge back to full model later

**Usage:**

```bash
python finetune_starwars.py --use-lora
```

Or in config:

```python
use_lora: bool = True
lora_r: int = 8              # LoRA rank (higher = more capacity)
lora_alpha: int = 16         # LoRA scaling factor
lora_dropout: float = 0.1    # Dropout for regularization
```

### Multi-GPU Training

The script automatically uses all available GPUs. To control GPU usage:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python finetune_starwars.py

# Use single GPU
CUDA_VISIBLE_DEVICES=0 python finetune_starwars.py
```

### Mixed Precision Training

Enabled by default on CUDA devices (`use_fp16: bool = True`). This:
- Reduces memory usage by ~50%
- Speeds up training
- Minimal impact on quality

### Data Augmentation

The dataset preparer includes simple augmentation:

```python
# In prepare_starwars_dataset.py
stories = preparer.augment_stories(stories)
```

This adds variations with different story starters to increase dataset diversity.

### Hyperparameter Tuning

Key parameters to adjust in `config/training_config.py`:

**Learning Rate** (`learning_rate`)
- Default: `5e-5`
- Too high: Training unstable, loss explodes
- Too low: Training very slow, may not converge
- Try: `3e-5` to `1e-4`

**Batch Size** (`batch_size` × `gradient_accumulation_steps`)
- Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
- Default: 4 × 4 = 16
- Larger batch sizes are more stable but require more memory
- Try: 8, 16, 32

**Number of Epochs** (`num_epochs`)
- Default: `3`
- Small datasets: 3-10 epochs
- Large datasets: 1-3 epochs
- Watch for overfitting!

**Max Length** (`max_length`)
- Default: `512` tokens
- Shorter sequences train faster
- Longer sequences allow longer stories
- TinyStories is trained on ~512 tokens

## Expected Results

### Training Time

On a single GPU (e.g., RTX 3090):
- **TinyStories-8M**: ~5-10 minutes for 3 epochs (small dataset)
- **TinyStories-33M**: ~15-30 minutes for 3 epochs (small dataset)
- With LoRA: ~30-50% faster

### Performance

After fine-tuning on Star Wars stories, the model should:
- Use Star Wars characters and locations
- Maintain simple, child-friendly language
- Follow story structure (beginning, middle, end)
- Stay coherent within the story

Example output:

```
Once upon a time, in a galaxy far, far away, a young Jedi named Luke
lived on a desert planet. One day, he found two droids in the sand.
The droids had a secret message from Princess Leia. She needed help
from a wise old Jedi named Obi-Wan Kenobi. Luke decided to go on an
adventure to save the princess and fight the evil Empire.
```

## Troubleshooting

### Out of Memory (OOM)

If training crashes with OOM errors:

1. **Reduce batch size**: `--batch-size 2` or `--batch-size 1`
2. **Increase gradient accumulation**: Keep effective batch size = batch_size × gradient_accumulation_steps
3. **Use LoRA**: `--use-lora`
4. **Reduce max length**: `--max-length 256`
5. **Use smaller model**: `--model roneneldan/TinyStories-8M`

### Poor Quality Generations

If the model generates nonsense:

1. **Train longer**: Increase `--epochs 5`
2. **Check your data**: Ensure stories are high quality
3. **Adjust temperature**: Lower temperature (0.7) for more coherent text
4. **More data**: Add more training examples (aim for 100+ stories)
5. **Check learning rate**: Try `--learning-rate 3e-5`

### Dataset Not Found

```
FileNotFoundError: Dataset not found at data/starwars_stories.json
```

Run the dataset preparation script first:

```bash
python prepare_starwars_dataset.py
```

## Integration with Attention Visualization

After fine-tuning, you can use the model with the existing attention visualization tools:

```python
from src.model_loader import AttentionExtractor
from src.visualizer import AttentionVisualizer

# Load your fine-tuned model
extractor = AttentionExtractor("models/tinystories-starwars/final")
visualizer = AttentionVisualizer(extractor)

# Generate and visualize
prompt = "Luke Skywalker met"
result = extractor.extract_activations(prompt)

# Plot attention patterns
visualizer.plot_all_heads(result['attention'], result['tokens'], layer=0)
```

Or use the Streamlit app:

```bash
streamlit run app.py
```

Then select "Custom Model" and enter: `models/tinystories-starwars/final`

## File Structure

```
show_me_your_attention/
├── config/
│   └── training_config.py          # Training configuration
├── data/
│   └── starwars_stories.json       # Training dataset
├── models/
│   └── tinystories-starwars/       # Fine-tuned model checkpoints
│       ├── checkpoint-100/
│       ├── checkpoint-200/
│       └── final/                  # Final model
├── prepare_starwars_dataset.py     # Dataset preparation
├── finetune_starwars.py            # Training script
└── FINETUNING.md                   # This file
```

## Tips for Creating Good Training Data

1. **Quality over quantity**: 50 high-quality stories > 500 poor ones
2. **Consistent style**: Use similar writing style and tone
3. **Appropriate length**: Stories should be 100-500 words
4. **Star Wars themed**: Include characters, locations, and concepts from Star Wars
5. **Child-friendly**: Simple language, clear narrative structure
6. **Diverse examples**: Cover different characters and scenarios

## Next Steps

1. Collect or generate more Star Wars stories
2. Fine-tune the model with your dataset
3. Evaluate generation quality
4. Iterate on hyperparameters
5. Use attention visualization to understand what the model learned

## Resources

- TinyStories paper: https://arxiv.org/abs/2305.07759
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- LoRA paper: https://arxiv.org/abs/2106.09685
- Star Wars Wookieepedia: https://starwars.fandom.com/

## Questions?

If you encounter issues or have questions about fine-tuning, check:
1. The error message for specific issues
2. Your dataset format and quality
3. GPU memory and compute resources
4. Training logs in `models/tinystories-starwars/logs/`