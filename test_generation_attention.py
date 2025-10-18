"""
Test script for the new generation attention feature.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_loader import AttentionExtractor
from visualizer import AttentionVisualizer

# Initialize
print("Loading model...")
extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")
viz = AttentionVisualizer()

# Test prompt
prompt = "Scare the cat."
target_word = "cat"

print(f"\nPrompt: {prompt}")
print(f"Target word: {target_word}")
print("\nGenerating text with attention tracking...")

# Generate with attention tracking
result = extractor.generate_with_attention_to_target(
    prompt=prompt,
    target_word=target_word,
    max_new_tokens=15,
    temperature=1.0
)

print(f"\nGenerated text: {result['generated_text']}")
print(f"\nGenerated tokens: {result['generated_tokens']}")
print(f"\nAttention scores: {result['attention_scores']}")

# Visualize
print("\nCreating visualization...")
viz.plot_generated_attention_to_target(
    generation_result=result,
    show_layer_breakdown=False
)

print("\n✅ Test completed successfully!")
