# Per-Word Attention Visualization

This feature allows you to visualize how each generated word attends to ALL words in the prompt, with a separate plot for each generated word.

## Quick Start

```bash
python test_per_word_attention.py
```

This will:
1. Load the TinyStories-33M model
2. Generate text from the prompt "The cat is"
3. Create a visualization showing attention from each generated word to all prompt words
4. Save the result to `outputs/test_per_word_attention.png`

## Usage in Your Code

```python
from model_loader import AttentionExtractor
from visualizer import AttentionVisualizer

# Initialize
extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
viz = AttentionVisualizer()

# Generate with attention tracking
result = extractor.generate_with_attention_to_all_prompt_tokens(
    prompt="Your prompt here",
    max_new_tokens=10,
    temperature=1.0
)

# Visualize
viz.plot_generated_attention_to_all_prompt_tokens(
    generation_result=result,
    save_path="output.png",
    figsize=(18, 12)  # Adjust size as needed
)
```

## What It Shows

- **Each subplot** represents one prompt token (stacked vertically)
- **X-axis**: Generated tokens (in order)
- **Y-axis**: Attention score (averaged across all layers and heads)
- **Bar height**: How much each generated token attends to that specific prompt token
- **Color**: Relative attention strength (darker = stronger attention within that subplot)

## Example Output

For prompt "The cat is" generating "scared. It runs away. The dog chases it."

You'll see 3 subplots (one for each prompt token):
1. **"The"** ← showing attention from all 12 generated tokens
2. **"cat"** ← showing attention from all 12 generated tokens
3. **"is"** ← showing attention from all 12 generated tokens

This layout makes it easy to see which generated words pay most attention to each specific prompt word.

## Differences from Original Visualization

**Original (`plot_generated_attention_to_target`):**
- Shows attention from all generated tokens to ONE specific target word
- Single plot with generated tokens on X-axis
- Used for tracking how generation relates to a specific important word

**New (`plot_generated_attention_to_all_prompt_tokens`):**
- Shows attention to EACH prompt token from ALL generated tokens
- Multiple subplots (one per prompt token, stacked vertically)
- Used for understanding which generated words attend most to each prompt word

## Files Added/Modified

### New Method in `model_loader.py`:
- `generate_with_attention_to_all_prompt_tokens()` - Generates text and tracks attention to all prompt tokens

### New Method in `visualizer.py`:
- `plot_generated_attention_to_all_prompt_tokens()` - Creates grid visualization

### Example Scripts:
- `test_per_word_attention.py` - Simple test script
- `examples/generation_attention_per_word.py` - Full demonstration

## Tips

1. **Adjust figure size**: Use `figsize=(width, height)` parameter for better readability
2. **Limit tokens**: Use `max_new_tokens` to control how many subplots are created
3. **Temperature**: Lower temperature (e.g., 0.7) for more focused generation
4. **Prompt length**: Works best with prompts of 3-10 tokens for clarity
