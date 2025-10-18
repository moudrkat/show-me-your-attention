"""Debug attention values to see what's actually happening."""

import sys
sys.path.insert(0, "src")

from model_loader import AttentionExtractor
import numpy as np

extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")

prompt = "Tell me about the unicorn"

result = extractor.extract_activations(prompt)
tokens = result["tokens"]

print(f"Prompt: {prompt}")
print(f"Tokens: {tokens}")

# Get attention from first layer
attn = result["attention_weights"][0][0].cpu().numpy()  # [num_heads, seq_len, seq_len]

print(f"\nAttention shape: {attn.shape}")
print(f"Number of heads: {attn.shape[0]}")
print(f"Sequence length: {attn.shape[1]}")

# Find 'tell' (index 0) and 'unicorn' (index 4)
tell_idx = 0
unicorn_idx = 4

print(f"\n'tell' token at index: {tell_idx}")
print(f"'unicorn' token at index: {unicorn_idx}")

print("\nAttention from 'tell' TO 'unicorn' across all heads:")
for head_idx in range(attn.shape[0]):
    attn_value = attn[head_idx, tell_idx, unicorn_idx]
    print(f"  Head {head_idx}: {attn_value:.6f}")

print("\nAttention from 'unicorn' TO 'tell' across all heads:")
for head_idx in range(attn.shape[0]):
    attn_value = attn[head_idx, unicorn_idx, tell_idx]
    print(f"  Head {head_idx}: {attn_value:.6f}")

print("\n\nFull attention matrix from head 0:")
print("From (rows) → To (columns)")
print(f"Shape: {attn[0].shape}")
print("\nHead 0 attention:")
for i, tok_from in enumerate(tokens):
    print(f"\n{tok_from:10s} → ", end="")
    for j, tok_to in enumerate(tokens):
        print(f"{attn[0, i, j]:.3f} ", end="")

# Now test the analyze function
print("\n\n" + "="*80)
print("Testing analyze_instruction_attention function:")
print("="*80)

results = extractor.analyze_instruction_attention(
    prompts=[prompt],
    target_word="unicorn",
    instruction_words=["tell"]
)

for prompt_key, data in results.items():
    print(f"\nPrompt: {prompt_key}")
    print(f"Instruction indices: {data['instruction_indices']}")
    print(f"Target indices: {data['target_indices']}")

    print("\nLayer 0 scores:")
    layer_0 = data['layer_scores']['layer_0']
    for key, value in layer_0.items():
        print(f"  {key}: {value:.6f}")