
"""
Analyze instruction-word attention linking across different prompt styles.
Tests whether adding emphasis words like "IMPORTANT" actually changes model attention.

This script compares funny/poetic instruction variations to see how models bind
instructions to target words at different layers.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_loader import AttentionExtractor
from visualizer import AttentionVisualizer


def test_emphasis_words():
    """
    Test if adding emphasis words like IMPORTANT, PLEASE, URGENT changes attention.
    Target word: "unicorn"
    """
    print("=" * 80)
    print("Testing Emphasis Words on Instruction Following")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    target_word = "unicorn"

    # Different levels of emphasis on the same instruction
    prompts = [
        "Tell me about the unicorn",
        "Please tell me about the unicorn",
        "IMPORTANT: Tell me about the unicorn",
        "URGENT!!! Tell me about the unicorn",
        "I beg you, tell me about the unicorn",
    ]

    instruction_words_sets = [
        ["tell"],  # Base instruction
        ["please", "tell"],  # Polite
        ["IMPORTANT", "tell"],  # Emphasized
        ["URGENT", "tell"],  # Very emphasized
        ["beg", "tell"],  # Dramatic
    ]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
        print(f"     Instruction words: {instruction_words_sets[i]}")

    print("\nAnalyzing instruction-word attention linking...")

    # Analyze each prompt with its specific instruction words
    all_results = {}
    for prompt, inst_words in zip(prompts, instruction_words_sets):
        print(f"\nProcessing: {prompt}")
        result = extractor.analyze_instruction_attention(
            prompts=[prompt],
            target_word=target_word,
            instruction_words=inst_words
        )
        all_results.update(result)

    # Visualize results
    print("\nGenerating visualizations...")

    viz.plot_instruction_attention_layers(
        analysis_results=all_results,
        target_word=target_word,
        save_path="outputs/emphasis_attention_layers.png"
    )

    # Use target_to_instruction since attention is causal!
    viz.plot_instruction_attention_heatmap(
        analysis_results=all_results,
        target_word=target_word,
        metric="target_to_instruction_mean",
        save_path="outputs/emphasis_attention_heatmap.png"
    )


def test_poetic_vs_direct():
    """
    Compare poetic/flowery instructions vs direct ones.
    Target word: "dragon"
    """
    print("\n" + "=" * 80)
    print("Testing Poetic vs Direct Instructions")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    target_word = "dragon"

    prompts = [
        "Describe the dragon",
        "Paint me a picture of the dragon",
        "Weave a tale about the dragon",
        "Illuminate the essence of the dragon",
        "The dragon needs describing",
    ]

    # These words get tokenized into subwords, so we use the first subword
    instruction_words = ["Desc", "aint", "ave", "Ill", "describing"]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")

    print(f"\nTarget word: '{target_word}'")
    print(f"Instruction words: {instruction_words}")

    print("\nAnalyzing instruction-word attention linking...")
    results = extractor.analyze_instruction_attention(
        prompts=prompts,
        target_word=target_word,
        instruction_words=instruction_words
    )

    # Print token information
    print("\nDetected tokens in each prompt:")
    for prompt, data in results.items():
        print(f"\n  Prompt: {prompt}")
        print(f"  Tokens: {data['tokens']}")
        print(f"  Instruction indices: {data['instruction_indices']}")
        print(f"  Target indices: {data['target_indices']}")

    print("\nGenerating visualizations...")

    viz.plot_instruction_attention_layers(
        analysis_results=results,
        target_word=target_word,
        save_path="outputs/poetic_attention_layers.png"
    )

    viz.plot_instruction_attention_heatmap(
        analysis_results=results,
        target_word=target_word,
        metric="target_to_instruction_mean",
        save_path="outputs/poetic_attention_heatmap.png"
    )


def test_word_order():
    """
    Test if word order affects instruction-target binding.
    Target word: "cat"
    """
    print("\n" + "=" * 80)
    print("Testing Word Order Effects")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    target_word = "cat"

    prompts = [
        "Focus on the cat",
        "The cat is what you should focus on",
        "On the cat, please focus",
        "Cat-focused analysis required",
    ]

    instruction_words = ["focus"]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")

    print(f"\nTarget word: '{target_word}'")
    print(f"Instruction words: {instruction_words}")

    print("\nAnalyzing instruction-word attention linking...")
    results = extractor.analyze_instruction_attention(
        prompts=prompts,
        target_word=target_word,
        instruction_words=instruction_words
    )

    print("\nGenerating visualizations...")

    viz.plot_instruction_attention_layers(
        analysis_results=results,
        target_word=target_word,
        save_path="outputs/word_order_attention_layers.png"
    )

    viz.plot_instruction_attention_heatmap(
        analysis_results=results,
        target_word=target_word,
        metric="target_to_instruction_mean",
        save_path="outputs/word_order_attention_heatmap.png"
    )


def test_dramatic_instructions():
    """
    Test absurdly dramatic instructions to see if they change anything.
    Target word: "butterfly"
    """
    print("\n" + "=" * 80)
    print("Testing Dramatically Absurd Instructions")
    print("=" * 80)

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
    viz = AttentionVisualizer()

    target_word = "butterfly"

    prompts = [
        "Describe the butterfly",
        "ATTENTION HUMAN: Describe the butterfly NOW",
        "By the power vested in me, DESCRIBE the butterfly",
        "Breaking news: Butterfly requires immediate description",
        "Dear model, with utmost respect, describe the butterfly",
    ]

    # "Describe" gets tokenized as "Desc" + "ribe", so we search for "Desc"
    instruction_words = ["Desc"]

    print("\nPrompts to compare:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")

    print(f"\nTarget word: '{target_word}'")
    print(f"Instruction words: {instruction_words}")

    print("\nAnalyzing instruction-word attention linking...")
    results = extractor.analyze_instruction_attention(
        prompts=prompts,
        target_word=target_word,
        instruction_words=instruction_words
    )

    print("\nGenerating visualizations...")

    viz.plot_instruction_attention_layers(
        analysis_results=results,
        target_word=target_word,
        save_path="outputs/dramatic_attention_layers.png"
    )

    viz.plot_instruction_attention_heatmap(
        analysis_results=results,
        target_word=target_word,
        metric="target_to_instruction_mean",
        save_path="outputs/dramatic_attention_heatmap.png"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze instruction-word attention linking with funny/poetic variations"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "emphasis", "poetic", "word_order", "dramatic"],
        help="Which test to run"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    if args.mode == "all":
        test_emphasis_words()
        test_poetic_vs_direct()
        test_word_order()
        test_dramatic_instructions()
    elif args.mode == "emphasis":
        test_emphasis_words()
    elif args.mode == "poetic":
        test_poetic_vs_direct()
    elif args.mode == "word_order":
        test_word_order()
    elif args.mode == "dramatic":
        test_dramatic_instructions()

    print("\n" + "=" * 80)
    print("Analysis complete! Check the outputs/ directory for visualizations.")
    print("\nInterpretation tips:")
    print("- Higher values = stronger instruction-target binding")
    print("- Look at middle layers (3-5) for instruction processing")
    print("- Compare 'Instruction → Target' to see if emphasis words help")
    print("=" * 80)
