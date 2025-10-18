"""
Quick test: Generate text and visualize attention from each generated word to all prompt words.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_loader import AttentionExtractor
from visualizer import AttentionVisualizer
import matplotlib.pyplot as plt

# Initialize
print("Loading model...")
extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
viz = AttentionVisualizer()

# Simple prompt
prompt = "The cat is"

print(f"\nPrompt: '{prompt}'")
print("Generating text...\n")

# Generate with attention tracking to all prompt tokens
result = extractor.generate_with_attention_to_all_prompt_tokens(
    prompt=prompt,
    max_new_tokens=12,
    temperature=1.0
)

print(f"Generated text: {result['generated_text']}")
print(f"\nPrompt tokens: {result['prompt_tokens']}")
print(f"Generated tokens: {result['generated_tokens']}")
print(f"\nCreating visualization...")

# Visualize - creates one subplot per generated token
viz.plot_generated_attention_to_all_prompt_tokens(
    generation_result=result,
    save_path="outputs/test_per_word_attention.png",
    figsize=(18, 12)
)

print("\n✅ Saved visualization to: outputs/test_per_word_attention.png")
print("\nEach subplot shows one prompt word and which generated words attend to it.")
