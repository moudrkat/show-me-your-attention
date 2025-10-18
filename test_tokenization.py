"""
Test the automatic tokenization handling for multi-token words.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_loader import AttentionExtractor

def test_tokenization():
    """Test that word finding works correctly even when words split into multiple tokens."""

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")

    # Test prompts
    prompts = [
        "Scare the cat",
        "I beg you, scare the cat",
        "IMPORTANT: scare the cat"
    ]

    print("=" * 80)
    print("Testing Automatic Tokenization Handling")
    print("=" * 80)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        # Tokenize to see what we're working with
        activations = extractor.extract_activations(prompt)
        tokens = activations["tokens"]
        print(f"Tokens: {tokens}")

        # Test finding "scare" (might be split as ["Sc", "are"])
        scare_indices = extractor._find_word_token_indices("scare", prompt)
        print(f"Indices for 'scare': {scare_indices}")
        print(f"  -> Tokens: {[tokens[i] for i in scare_indices]}")

        # Test finding "cat" (should NOT match "scare" due to word boundaries)
        cat_indices = extractor._find_word_token_indices("cat", prompt)
        print(f"Indices for 'cat': {cat_indices}")
        print(f"  -> Tokens: {[tokens[i] for i in cat_indices]}")

    print("\n" + "=" * 80)
    print("Testing Full Analysis")
    print("=" * 80)

    # Now test the full analysis
    results = extractor.analyze_instruction_attention(
        prompts=prompts,
        target_word="cat",
        instruction_words=["scare"]  # Full word, not "Sc"!
    )

    print(f"\nAnalyzed {len(results)} prompts successfully!")

    for prompt, data in results.items():
        print(f"\nPrompt: {prompt}")
        print(f"  Tokens: {data['tokens']}")
        print(f"  Instruction indices (scare): {data['instruction_indices']}")
        print(f"  Target indices (cat): {data['target_indices']}")

        # Show layer 0 attention
        layer_0 = data['layer_scores']['layer_0']
        print(f"  Layer 0 target->instruction mean: {layer_0['target_to_instruction_mean']:.4f}")

if __name__ == "__main__":
    test_tokenization()
