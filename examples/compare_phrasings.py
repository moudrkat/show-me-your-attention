

"""
Example script demonstrating how different prompt phrasings affect attention patterns.
Compares similar prompts with different wording to visualize differences in model internals.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_loader import AttentionExtractor
from visualizer import AttentionVisualizer
import matplotlib.pyplot as plt


def compare_simple_phrasings():
    """
    Compare attention patterns for simple phrasings of the same concept.
    """
    print("=" * 80)
    print("Comparing Simple Phrasings")
    print("=" * 80)

    # Initialize model and visualizer
    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    # Define different phrasings of similar concepts
    prompts = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",
        "On the mat, there sat a cat",
    ]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    # Extract activations for all prompts
    print("\nExtracting activations...")
    results = extractor.compare_prompts(prompts)

    # Visualize attention for each prompt (first layer, first head)
    print("\nGenerating attention heatmaps...")
    for prompt in prompts:
        result = results[prompt]
        viz.plot_attention_heatmap(
            attention_weights=result["attention_weights"][0],
            tokens=result["tokens"],
            layer_idx=0,
            head_idx=0,
            save_path=f"outputs/attention_{prompts.index(prompt)}.png",
            prompt=prompt
        )

    # Compare neuron activations across prompts
    print("\nComparing neuron activations...")
    activations_dict = {}
    for prompt in prompts:
        result = results[prompt]
        # Get last layer hidden states
        activations_dict[prompt] = result["hidden_states"][-1].squeeze(0).cpu().numpy()

    viz.plot_activation_comparison(
        activations_dict=activations_dict,
        layer_idx=extractor.model.config.num_hidden_layers - 1,
        save_path="outputs/neuron_comparison.png"
    )


def compare_question_forms():
    """
    Compare attention patterns for different question formulations.
    """
    print("\n" + "=" * 80)
    print("Comparing Question Forms")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    prompts = [
        "What is your name?",
        "Can you tell me your name?",
        "Your name is what?",
        "I want to know your name",
    ]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    print("\nExtracting activations...")
    results = extractor.compare_prompts(prompts)

    # Plot all attention heads for each prompt
    print("\nVisualizing all attention heads for each prompt...")
    for i, prompt in enumerate(prompts):
        result = results[prompt]
        viz.plot_all_heads(
            attention_weights=result["attention_weights"][0],
            tokens=result["tokens"],
            layer_idx=0,
            save_path=f"outputs/all_heads_question_{i}.png",
            prompt=prompt
        )

    # Compare attention statistics across layers
    print("\nComparing layer statistics...")
    stats = extractor.get_attention_stats(prompts[0])
    viz.plot_layer_statistics(
        stats_dict=stats,
        save_path="outputs/layer_stats.png"
    )


def compare_sentiment():
    """
    Compare attention patterns for different sentiment expressions.
    """
    print("\n" + "=" * 80)
    print("Comparing Sentiment Expressions")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    prompts = [
        "I am very happy today",
        "I am extremely sad today",
        "Today is a wonderful day",
        "Today is a terrible day",
    ]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    print("\nExtracting activations...")
    results = extractor.compare_prompts(prompts)

    # For each prompt, show neuron activations
    print("\nVisualizing neuron activations...")
    for prompt in prompts:
        result = results[prompt]
        neuron_acts = result["hidden_states"][-1].squeeze(0).cpu().numpy()

        viz.plot_neuron_activations(
            activations=neuron_acts,
            tokens=result["tokens"],
            layer_idx=extractor.model.config.num_hidden_layers - 1,
            top_k=30,
            save_path=f"outputs/neurons_sentiment_{prompts.index(prompt)}.png",
            prompt=prompt
        )


def interactive_exploration():
    """
    Create interactive visualizations for exploring attention patterns.
    """
    print("\n" + "=" * 80)
    print("Creating Interactive Visualizations")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    prompt = "Once upon a time in a magical forest"

    print(f"\nPrompt: {prompt}")
    print("Extracting activations...")

    result = extractor.extract_activations(prompt)

    print("Creating interactive attention visualization...")
    viz.plot_interactive_attention(
        attention_weights=result["attention_weights"][0],
        tokens=result["tokens"],
        layer_idx=0,
        save_path="outputs/interactive_attention.html"
    )

    print("\nInteractive visualization saved to outputs/interactive_attention.html")
    print("Open this file in a web browser to explore!")


def custom_comparison(prompt1: str, prompt2: str):
    """
    Compare two custom prompts.

    Args:
        prompt1: First prompt
        prompt2: Second prompt
    """
    print("\n" + "=" * 80)
    print("Custom Prompt Comparison")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    prompts = [prompt1, prompt2]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    print("\nExtracting activations...")
    results = extractor.compare_prompts(prompts)

    # Show attention for both
    for i, prompt in enumerate(prompts):
        result = results[prompt]
        viz.plot_attention_heatmap(
            attention_weights=result["attention_weights"][0],
            tokens=result["tokens"],
            layer_idx=0,
            head_idx=0,
            save_path=f"outputs/custom_attention_{i}.png",
            prompt=prompt
        )

    # Compare neuron activations
    activations_dict = {}
    for prompt in prompts:
        result = results[prompt]
        activations_dict[prompt] = result["hidden_states"][-1].squeeze(0).cpu().numpy()

    viz.plot_activation_comparison(
        activations_dict=activations_dict,
        layer_idx=extractor.model.config.num_hidden_layers - 1,
        save_path="outputs/custom_neuron_comparison.png"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare attention patterns across different prompt phrasings")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "simple", "questions", "sentiment", "interactive", "custom"],
                        help="Which comparison to run")
    parser.add_argument("--prompt1", type=str, help="First prompt for custom comparison")
    parser.add_argument("--prompt2", type=str, help="Second prompt for custom comparison")

    args = parser.parse_args()

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    if args.mode == "all":
        compare_simple_phrasings()
        compare_question_forms()
        compare_sentiment()
        interactive_exploration()
    elif args.mode == "simple":
        compare_simple_phrasings()
    elif args.mode == "questions":
        compare_question_forms()
    elif args.mode == "sentiment":
        compare_sentiment()
    elif args.mode == "interactive":
        interactive_exploration()
    elif args.mode == "custom":
        if not args.prompt1 or not args.prompt2:
            print("Error: --prompt1 and --prompt2 required for custom mode")
            sys.exit(1)
        custom_comparison(args.prompt1, args.prompt2)

    print("\n" + "=" * 80)
    print("Done! Check the outputs/ directory for visualizations.")
    print("=" * 80)
