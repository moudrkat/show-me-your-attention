# Show Me Your Attention

A Python project for visualizing neuron activations and attention patterns from transformer-based language models. Load tiny LLMs from HuggingFace and explore how different prompt phrasings affect internal model representations.

## Features

- **Model Loading**: Easy loading of small LLMs from HuggingFace Hub
- **Attention Extraction**: Capture attention weights from all layers and heads
- **Neuron Activations**: Extract and analyze hidden state activations
- **Rich Visualizations**: Multiple visualization types including:
  - Attention heatmaps (single head or all heads)
  - Neuron activation patterns
  - Interactive Plotly visualizations
  - Cross-prompt comparisons
  - Layer-wise statistics

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd show_me_your_attention
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.model_loader import AttentionExtractor
from src.visualizer import AttentionVisualizer

# Load a tiny model (33M parameters)
extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")

# Extract activations for a prompt
result = extractor.extract_activations("Once upon a time")

# Visualize attention patterns
viz = AttentionVisualizer()
viz.plot_attention_heatmap(
    attention_weights=result["attention_weights"][0],
    tokens=result["tokens"],
    layer_idx=0,
    head_idx=0
)
```

### Running Examples

The `examples/` directory contains ready-to-use scripts:

```bash
# Run all comparisons
python examples/compare_phrasings.py --mode all

# Compare simple phrasings
python examples/compare_phrasings.py --mode simple

# Compare question formulations
python examples/compare_phrasings.py --mode questions

# Compare sentiment expressions
python examples/compare_phrasings.py --mode sentiment

# Create interactive visualization
python examples/compare_phrasings.py --mode interactive

# Compare custom prompts
python examples/compare_phrasings.py --mode custom \
    --prompt1 "The cat sat on the mat" \
    --prompt2 "A cat was sitting on a mat"
```

## Project Structure

```
show_me_your_attention/
├── src/
│   ├── __init__.py
│   ├── model_loader.py      # Model loading and activation extraction
│   └── visualizer.py         # Visualization tools
├── examples/
│   └── compare_phrasings.py  # Example comparison scripts
├── outputs/                  # Generated visualizations (auto-created)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Available Models

The project works with any HuggingFace transformer model that supports `output_attentions=True`. Some recommended tiny models:

- **roneneldan/TinyStories-33M** (default) - 33M parameters, trained on children's stories
- **roneneldan/TinyStories-8M** - 8M parameters, even smaller
- **gpt2** - 124M parameters, classic GPT-2 small
- **distilgpt2** - 82M parameters, distilled version of GPT-2

To use a different model:
```python
extractor = AttentionExtractor(model_name="distilgpt2")
```

## API Reference

### AttentionExtractor

Main class for loading models and extracting activations.

**Methods:**
- `extract_activations(prompt: str)` - Extract all activations for a prompt
- `get_attention_stats(prompt: str)` - Get statistical summary per layer
- `compare_prompts(prompts: List[str])` - Compare multiple prompts
- `get_neuron_activations(prompt: str, layer_idx: int)` - Get specific layer activations
- `get_model_info()` - Get model architecture information

### AttentionVisualizer

Visualization tools for attention patterns and activations.

**Methods:**
- `plot_attention_heatmap()` - Single attention head heatmap
- `plot_all_heads()` - Grid of all attention heads
- `plot_neuron_activations()` - Neuron activation patterns
- `plot_activation_comparison()` - Compare activations across prompts
- `plot_interactive_attention()` - Interactive Plotly visualization
- `plot_layer_statistics()` - Layer-wise attention statistics

## Example Outputs

The scripts generate various visualizations in the `outputs/` directory:

- `attention_*.png` - Attention heatmaps for individual prompts
- `all_heads_*.png` - Grid showing all attention heads
- `neuron_comparison.png` - Neuron activation comparison across prompts
- `layer_stats.png` - Statistical summary across layers
- `interactive_attention.html` - Interactive exploration tool

## Understanding the Visualizations

### Attention Heatmaps
- **Rows (Query)**: Source tokens asking "what should I attend to?"
- **Columns (Key)**: Target tokens being attended to
- **Brighter colors**: Stronger attention weights
- Each head in each layer learns different patterns

### Neuron Activations
- Shows how individual neurons respond to each token
- Top-k most variable neurons are displayed
- Red/blue colors show positive/negative activations
- Useful for understanding what features neurons detect

### Comparison Plots
- Line plots showing how specific neurons activate differently across prompts
- Helps identify which neurons are sensitive to phrasing changes

## Advanced Usage

### Custom Analysis

```python
# Get model architecture info
info = extractor.get_model_info()
print(f"Model has {info['num_layers']} layers")
print(f"Total parameters: {info['total_parameters']:,}")

# Get statistics for a prompt
stats = extractor.get_attention_stats("Your prompt here")
for layer, metrics in stats.items():
    print(f"{layer}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}")

# Extract specific layer activations
layer_activations = extractor.get_neuron_activations(
    "Your prompt",
    layer_idx=-1  # Last layer
)
print(f"Activation shape: {layer_activations.shape}")
```

### Batch Processing

```python
prompts = [
    "Prompt 1",
    "Prompt 2",
    "Prompt 3"
]

results = extractor.compare_prompts(prompts)

# Analyze differences
for prompt, data in results.items():
    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {len(data['tokens'])}")
    print(f"Layers: {len(data['attention_weights'])}")
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- matplotlib, seaborn, plotly
- numpy

See `requirements.txt` for full list.

## Tips for Exploration

1. **Start small**: Use TinyStories-8M or 33M for fast iteration
2. **Focus on specific heads**: Not all attention heads are equally interpretable
3. **Compare similar prompts**: Small changes often reveal interesting patterns
4. **Look at multiple layers**: Early layers capture syntax, later layers capture semantics
5. **Use interactive visualizations**: The HTML outputs allow zooming and exploration

## Contributing

Feel free to open issues or submit pull requests with improvements:
- Additional visualization types
- Support for more model architectures
- Analysis tools and metrics
- Example notebooks

## License

MIT License - feel free to use this code for research and education.

## Acknowledgments

- Built on HuggingFace Transformers
- Default model: TinyStories by Eldan & Li (2023)
- Visualization inspired by BertViz and similar tools

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{show_me_your_attention,
  title={Show Me Your Attention: Visualizing LLM Attention Patterns},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/show_me_your_attention}
}
```

---

Happy exploring! Watch how models think differently based on how you phrase things.
