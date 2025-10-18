"""Quick debug script to see how tokens are actually split."""

import sys
sys.path.insert(0, "src")

from model_loader import AttentionExtractor

extractor = AttentionExtractor(model_name="roneneldan/TinyStories-33M")

prompts = [
    "Tell me about the unicorn",
    "IMPORTANT: Tell me about the unicorn",
]

for prompt in prompts:
    result = extractor.extract_activations(prompt)
    tokens = result["tokens"]

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {tokens}")
    print("\nToken index mapping:")
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace('Ġ', '_').replace('▁', '_')
        print(f"  {i}: '{clean_tok}'")

    # Try to find 'tell' and 'unicorn'
    print("\nLooking for 'tell':")
    for i, tok in enumerate(tokens):
        if 'tell' in tok.lower().replace('Ġ', '').replace('▁', ''):
            print(f"  Found at index {i}: {tok}")

    print("\nLooking for 'unicorn':")
    for i, tok in enumerate(tokens):
        if 'unicorn' in tok.lower().replace('Ġ', '').replace('▁', ''):
            print(f"  Found at index {i}: {tok}")
