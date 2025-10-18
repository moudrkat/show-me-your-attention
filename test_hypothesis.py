"""
Test hypothesis: Higher attention = better topic coherence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_loader import AttentionExtractor

def test_attention_vs_coherence():
    """
    Test if higher target→instruction attention correlates with staying on topic.
    """

    extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")

    prompts = [
        "Scare the cat",
        "I beg you, scare the cat",
        "IMPORTANT: scare the cat"
    ]

    print("=" * 80)
    print("HYPOTHESIS TEST: Does Higher Attention = Better Topic Coherence?")
    print("=" * 80)

    # Get attention scores
    results = extractor.analyze_instruction_attention(
        prompts=prompts,
        target_word="cat",
        instruction_words=["scare"]
    )

    # Get generated text for each
    generations = {}
    for prompt in prompts:
        generations[prompt] = extractor.generate_text(prompt, max_length=50)

    # Show results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Calculate average attention across all layers
    attention_scores = {}
    for prompt, data in results.items():
        scores = []
        for layer_idx in range(len(data['layer_scores'])):
            layer_key = f"layer_{layer_idx}"
            score = data['layer_scores'][layer_key]['target_to_instruction_mean']
            scores.append(score)

        avg_attention = sum(scores) / len(scores)
        attention_scores[prompt] = avg_attention

    # Sort by attention (highest first)
    sorted_prompts = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nRanked by Average Target→Instruction Attention:\n")

    for rank, (prompt, attention) in enumerate(sorted_prompts, 1):
        print(f"{rank}. ATTENTION SCORE: {attention:.4f}")
        print(f"   Prompt: \"{prompt}\"")
        print(f"   Generated: \"{generations[prompt]}\"")
        print()

        # Analysis: Does it stay on topic?
        generated_lower = generations[prompt].lower()
        has_scare = "scare" in generated_lower or "scared" in generated_lower
        has_cat = "cat" in generated_lower

        print(f"   📊 Topic coherence:")
        print(f"      - Mentions 'scare/scared': {'✅ YES' if has_scare else '❌ NO'}")
        print(f"      - Mentions 'cat': {'✅ YES' if has_cat else '❌ NO'}")
        print(f"      - Stays on topic: {'✅ YES' if (has_scare and has_cat) else '❌ NO'}")
        print()
        print("-" * 80)
        print()

    # Summary
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    best_prompt = sorted_prompts[0][0]
    best_gen = generations[best_prompt]

    print(f"\nPrompt with HIGHEST attention: \"{best_prompt}\"")
    print(f"Does it stay most on-topic? Let's see...")
    print(f"\nGenerated: {best_gen}")
    print()

if __name__ == "__main__":
    test_attention_vs_coherence()
