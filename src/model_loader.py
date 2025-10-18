"""
Model loader with attention extraction capabilities.
Loads tiny LLMs from HuggingFace and captures attention layer activations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Dict, List, Tuple, Optional
import numpy as np


class AttentionExtractor:
    """
    Wrapper for HuggingFace models that extracts attention activations.
    Supports both causal (GPT-style) and bidirectional (BERT-style) models.
    """

    def __init__(self, model_name: str = "roneneldan/TinyStories-33M", device: str = "auto"):
        """
        Initialize the model and tokenizer.

        Args:
            model_name: HuggingFace model identifier (default: TinyStories-33M, a 33M parameter model)
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Detect model type and load appropriately
        self.is_causal = self._is_causal_model(model_name)

        if self.is_causal:
            print("Detected: Causal (autoregressive) model")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True
            ).to(self.device)
        else:
            print("Detected: Bidirectional (masked) model")
            self.model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True
            ).to(self.device)

        self.model.eval()

        # Storage for activations
        self.attention_weights = []
        self.hidden_states = []

    def _is_causal_model(self, model_name: str) -> bool:
        """
        Detect if model is causal (GPT-style) or bidirectional (BERT-style).

        Args:
            model_name: HuggingFace model identifier

        Returns:
            True if causal, False if bidirectional
        """
        causal_keywords = ['gpt', 'tinystories', 'llama', 'opt', 'bloom', 'falcon']
        bidirectional_keywords = ['bert', 'roberta', 'albert', 'electra', 'distilbert']

        name_lower = model_name.lower()

        for keyword in bidirectional_keywords:
            if keyword in name_lower:
                return False

        for keyword in causal_keywords:
            if keyword in name_lower:
                return True

        # Default to causal if unsure
        return True

    def extract_activations(self, prompt: str) -> Dict:
        """
        Run the model on a prompt and extract attention activations.

        Args:
            prompt: Input text prompt

        Returns:
            Dictionary containing:
                - tokens: List of token strings
                - token_ids: Tensor of token IDs
                - attention_weights: List of attention weight tensors (one per layer)
                - hidden_states: List of hidden state tensors (one per layer)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Run forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract attention weights (tuple of tensors, one per layer)
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attention_weights = outputs.attentions

        # Extract hidden states (tuple of tensors, one per layer + embedding)
        # Shape: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.hidden_states

        return {
            "tokens": tokens,
            "token_ids": inputs["input_ids"],
            "attention_weights": attention_weights,
            "hidden_states": hidden_states,
            "logits": outputs.logits
        }

    def get_attention_stats(self, prompt: str) -> Dict:
        """
        Get statistical summary of attention patterns for a prompt.

        Args:
            prompt: Input text prompt

        Returns:
            Dictionary with attention statistics per layer
        """
        activations = self.extract_activations(prompt)
        stats = {}

        for layer_idx, attn in enumerate(activations["attention_weights"]):
            # Average across heads and batch
            attn_np = attn.squeeze(0).cpu().numpy()  # [num_heads, seq_len, seq_len]

            stats[f"layer_{layer_idx}"] = {
                "mean": float(attn_np.mean()),
                "std": float(attn_np.std()),
                "max": float(attn_np.max()),
                "min": float(attn_np.min()),
                "shape": attn_np.shape
            }

        return stats

    def compare_prompts(self, prompts: List[str]) -> Dict:
        """
        Compare attention patterns across multiple prompts.

        Args:
            prompts: List of text prompts to compare

        Returns:
            Dictionary mapping each prompt to its activations
        """
        results = {}

        for prompt in prompts:
            print(f"Processing: {prompt[:50]}...")
            results[prompt] = self.extract_activations(prompt)

        return results

    def get_neuron_activations(self, prompt: str, layer_idx: int = -1) -> np.ndarray:
        """
        Get neuron activations for a specific layer.

        Args:
            prompt: Input text prompt
            layer_idx: Layer index (-1 for last layer)

        Returns:
            Numpy array of neuron activations [seq_len, hidden_size]
        """
        activations = self.extract_activations(prompt)
        hidden_state = activations["hidden_states"][layer_idx]
        return hidden_state.squeeze(0).cpu().numpy()

    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text continuation from the model.

        Args:
            prompt: Input prompt to continue
            max_length: Maximum total length (prompt + generation)
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text string
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate without forcing attention outputs (they interfere with generation)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                return_dict_in_generate=False  # Return plain tensor, not GenerateOutput object
            )

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    def generate_with_attention_to_target(
        self,
        prompt: str,
        target_word: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        layer_idx: int = -1
    ) -> Dict:
        """
        Generate text while tracking attention from each generated token to a target word.

        Args:
            prompt: Input prompt
            target_word: Target word in the prompt to track attention to
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            layer_idx: Which layer to track (-1 for last layer, or average across all)

        Returns:
            Dictionary containing:
                - generated_text: Full generated text
                - prompt_tokens: List of prompt token strings
                - generated_tokens: List of generated token strings
                - target_indices: Indices of target word in prompt
                - attention_scores: List of attention scores (one per generated token)
                - layer_attention_scores: Dict mapping layer -> attention scores (if layer_idx == -1)
        """
        # Find target word indices
        target_indices = self._find_word_token_indices(target_word, prompt)
        if not target_indices:
            raise ValueError(f"Target word '{target_word}' not found in prompt")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_token_ids = inputs["input_ids"][0]
        prompt_tokens = self.tokenizer.convert_ids_to_tokens(prompt_token_ids)

        generated_tokens = []
        attention_scores = []
        layer_attention_scores = {f"layer_{i}": [] for i in range(self.model.config.num_hidden_layers)}

        # Generate token by token
        current_ids = prompt_token_ids.unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass with attention outputs
                outputs = self.model(
                    input_ids=current_ids,
                    output_attentions=True,
                    return_dict=True
                )

                # Get logits and sample next token
                logits = outputs.logits[0, -1, :]  # Last token's logits

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                # Get the token string
                next_token = self.tokenizer.decode(next_token_id)
                generated_tokens.append(next_token)

                # Extract attention from the newly generated token (last position) to target word
                # outputs.attentions is a tuple of [batch, num_heads, seq_len, seq_len]
                new_token_pos = current_ids.shape[1] - 1  # Position of the newly generated token

                # Calculate attention to target for each layer
                for layer_i, attn_layer in enumerate(outputs.attentions):
                    attn = attn_layer[0].cpu().numpy()  # [num_heads, seq_len, seq_len]

                    # Average attention from last token to all target indices across all heads
                    attn_to_target = []
                    for head_idx in range(attn.shape[0]):
                        for tgt_idx in target_indices:
                            attn_to_target.append(attn[head_idx, new_token_pos, tgt_idx])

                    avg_attn = float(np.mean(attn_to_target)) if attn_to_target else 0.0
                    layer_attention_scores[f"layer_{layer_i}"].append(avg_attn)

                # Calculate average attention across all layers or use specific layer
                if layer_idx == -1:
                    # Average across all layers
                    avg_across_layers = np.mean([layer_attention_scores[f"layer_{i}"][-1]
                                                 for i in range(len(outputs.attentions))])
                    attention_scores.append(float(avg_across_layers))
                else:
                    # Use specific layer
                    attention_scores.append(layer_attention_scores[f"layer_{layer_idx}"][-1])

                # Append new token to sequence
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)

                # Check for EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        # Decode full generated text
        generated_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)

        return {
            "generated_text": generated_text,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "target_word": target_word,
            "target_indices": target_indices,
            "attention_scores": attention_scores,
            "layer_attention_scores": layer_attention_scores
        }

    def generate_with_attention_to_all_prompt_tokens(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        layer_mode: str = "last"
    ) -> Dict:
        """
        Generate text while tracking attention from each generated token to ALL prompt tokens.

        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            layer_mode: Which layers to use for attention calculation:
                - "last": Use only the last layer (default)
                - "average": Average across all layers
                - "layer_N": Use specific layer N (e.g., "layer_0", "layer_7")

        Returns:
            Dictionary containing:
                - generated_text: Full generated text
                - prompt: Original prompt
                - prompt_tokens: List of prompt token strings
                - generated_tokens: List of generated token strings
                - attention_to_prompt: List of arrays, each array is [num_prompt_tokens] showing
                  attention from that generated token to each prompt token (averaged across layers and heads)
                - token_probabilities: List of probabilities for each generated token
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_token_ids = inputs["input_ids"][0]
        prompt_tokens = self.tokenizer.convert_ids_to_tokens(prompt_token_ids)
        num_prompt_tokens = len(prompt_token_ids)

        generated_tokens = []
        attention_to_prompt = []  # List of [num_prompt_tokens] arrays
        token_probabilities = []  # List of probabilities for each generated token

        # Generate token by token
        current_ids = prompt_token_ids.unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass with attention outputs
                outputs = self.model(
                    input_ids=current_ids,
                    output_attentions=True,
                    return_dict=True
                )

                # Get logits and sample next token
                logits = outputs.logits[0, -1, :]  # Last token's logits

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                # Store the probability of the sampled token
                token_prob = probs[next_token_id].item()
                token_probabilities.append(token_prob)

                # Get the token string
                next_token = self.tokenizer.decode(next_token_id)
                generated_tokens.append(next_token)

                # Extract attention from the newly generated token (last position) to ALL prompt tokens
                new_token_pos = current_ids.shape[1] - 1  # Position of the newly generated token

                # Calculate attention to each prompt token based on layer_mode
                attention_to_each_prompt_token = np.zeros(num_prompt_tokens)

                # Determine which layers to use
                num_layers = len(outputs.attentions)
                if layer_mode == "last":
                    layers_to_use = [num_layers - 1]
                elif layer_mode == "average":
                    layers_to_use = list(range(num_layers))
                elif layer_mode.startswith("layer_"):
                    try:
                        layer_num = int(layer_mode.split("_")[1])
                        if 0 <= layer_num < num_layers:
                            layers_to_use = [layer_num]
                        else:
                            layers_to_use = [num_layers - 1]  # Fallback to last layer
                    except:
                        layers_to_use = [num_layers - 1]  # Fallback to last layer
                else:
                    layers_to_use = [num_layers - 1]  # Default to last layer

                # Accumulate attention from selected layers
                for layer_i in layers_to_use:
                    attn_layer = outputs.attentions[layer_i]
                    attn = attn_layer[0].cpu().numpy()  # [num_heads, seq_len, seq_len]

                    # Average across all heads for this layer
                    for head_idx in range(attn.shape[0]):
                        for prompt_idx in range(num_prompt_tokens):
                            attention_to_each_prompt_token[prompt_idx] += attn[head_idx, new_token_pos, prompt_idx]

                # Average by number of layers and heads used
                num_heads = outputs.attentions[0].shape[1]
                attention_to_each_prompt_token /= (len(layers_to_use) * num_heads)

                attention_to_prompt.append(attention_to_each_prompt_token)

                # Append new token to sequence
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)

                # Check for EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        # Decode full generated text
        generated_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)

        return {
            "generated_text": generated_text,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "attention_to_prompt": attention_to_prompt,
            "token_probabilities": token_probabilities
        }

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model architecture.

        Returns:
            Dictionary with model configuration details
        """
        config = self.model.config

        return {
            "model_name": self.model_name,
            "model_type": "Causal (Autoregressive)" if self.is_causal else "Bidirectional (Masked)",
            "is_causal": self.is_causal,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": getattr(config, "max_position_embeddings", "N/A"),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def _find_word_token_indices(self, word: str, prompt: str) -> List[int]:
        """
        Find ALL token indices that correspond to a word in the prompt.
        Handles multi-token words (e.g., "scare" -> ["Sc", "are"]).
        Uses tokenizer's offset_mapping for accurate character-to-token alignment.

        Args:
            word: The word to find (e.g., "scare", "cat")
            prompt: The original prompt string

        Returns:
            List of token indices that make up this word
        """
        import re

        # Normalize the word
        word_lower = word.lower().strip()

        # Find word boundaries in original prompt (whole word only)
        prompt_lower = prompt.lower()
        word_positions = []

        for match in re.finditer(r'\b' + re.escape(word_lower) + r'\b', prompt_lower):
            word_positions.append((match.start(), match.end()))

        if not word_positions:
            return []

        # Tokenize with offset mapping to get character spans
        encoding = self.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        offset_mapping = encoding["offset_mapping"]

        # Find which tokens overlap with word positions
        token_indices = []
        for token_idx, (start, end) in enumerate(offset_mapping):
            # Check if this token overlaps with any occurrence of the word
            for word_start, word_end in word_positions:
                # Token overlaps if: token_start < word_end AND token_end > word_start
                if start < word_end and end > word_start:
                    if token_idx not in token_indices:
                        token_indices.append(token_idx)
                    break

        return sorted(token_indices)

    def extract_qkv_matrices(
        self,
        prompt: str,
        layer_idx: int = -1
    ) -> Dict:
        """
        Extract Query, Key, Value matrices for a given prompt at a specific layer.

        Args:
            prompt: Input text prompt
            layer_idx: Which layer to extract from (-1 for last layer)

        Returns:
            Dictionary containing:
                - tokens: List of token strings
                - Q: Query matrix [num_heads, seq_len, head_dim]
                - K: Key matrix [num_heads, seq_len, head_dim]
                - V: Value matrix [num_heads, seq_len, head_dim]
                - layer_idx: The actual layer index used
        """
        # Storage for Q, K, V
        qkv_storage = {}

        def hook_fn(module, input, output):
            """Hook to capture Q, K, V from attention layer"""
            hidden_states = input[0]
            batch_size, seq_len, _ = hidden_states.shape

            # Get Q, K, V projections
            query = module.q_proj(hidden_states)
            key = module.k_proj(hidden_states)
            value = module.v_proj(hidden_states)

            # Reshape to [batch, num_heads, seq_len, head_dim]
            num_heads = module.num_heads
            head_dim = module.head_dim

            query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            qkv_storage['Q'] = query[0].detach().cpu().numpy()  # [num_heads, seq_len, head_dim]
            qkv_storage['K'] = key[0].detach().cpu().numpy()
            qkv_storage['V'] = value[0].detach().cpu().numpy()

        # Determine which layer to hook
        num_layers = self.model.config.num_hidden_layers
        if layer_idx == -1:
            layer_idx = num_layers - 1
        elif layer_idx < 0 or layer_idx >= num_layers:
            layer_idx = num_layers - 1

        # Register hook on the specific layer's attention module
        if hasattr(self.model, 'transformer'):  # GPT-style models
            attention_module = self.model.transformer.h[layer_idx].attn.attention
        elif hasattr(self.model, 'encoder'):  # BERT-style models
            attention_module = self.model.encoder.layer[layer_idx].attention.self
        else:
            raise ValueError("Unknown model architecture")

        hook = attention_module.register_forward_hook(hook_fn)

        # Tokenize and run forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            _ = self.model(**inputs)

        # Remove hook
        hook.remove()

        return {
            "tokens": tokens,
            "Q": qkv_storage['Q'],
            "K": qkv_storage['K'],
            "V": qkv_storage['V'],
            "layer_idx": layer_idx,
            "num_heads": attention_module.num_heads,
            "head_dim": attention_module.head_dim
        }

    def analyze_instruction_attention(
        self,
        prompts: List[str],
        target_word: str,
        instruction_words: List[str]
    ) -> Dict:
        """
        Analyze how instruction tokens attend to a target word across different prompts.
        This measures whether the model is "linking" instructions to the target word.
        Automatically handles multi-token words.

        Args:
            prompts: List of prompts to compare (should contain target_word and instruction_words)
            target_word: The word that should be the focus of the instruction (e.g., "cat", "scare")
            instruction_words: Words that represent the instruction (e.g., ["describe", "IMPORTANT"])

        Returns:
            Dictionary with analysis results:
                - per_prompt: Dict mapping prompt to attention scores per layer
                - summary: Aggregated statistics
        """
        results = {}

        for prompt in prompts:
            # Extract activations
            activations = self.extract_activations(prompt)
            tokens = activations["tokens"]

            # Find indices using smart word-boundary matching with offset mapping
            instruction_indices = []
            for inst_word in instruction_words:
                indices = self._find_word_token_indices(inst_word, prompt)
                instruction_indices.extend(indices)

            # Remove duplicates and sort
            instruction_indices = sorted(list(set(instruction_indices)))

            target_indices = self._find_word_token_indices(target_word, prompt)

            if not instruction_indices or not target_indices:
                print(f"Warning: Could not find instruction or target word in: {prompt}")
                continue

            # Calculate attention from instruction tokens to target word for each layer
            layer_scores = {}
            layer_attention_patterns = {}

            for layer_idx, attn_layer in enumerate(activations["attention_weights"]):
                # attn_layer shape: [batch, num_heads, seq_len, seq_len]
                attn = attn_layer[0].cpu().numpy()  # [num_heads, seq_len, seq_len]

                # Collect attention from instruction tokens TO target word
                attention_to_target = []
                attention_from_target = []

                for head_idx in range(attn.shape[0]):
                    for inst_idx in instruction_indices:
                        for tgt_idx in target_indices:
                            # Attention FROM instruction TO target
                            attention_to_target.append(attn[head_idx, inst_idx, tgt_idx])
                            # Attention FROM target TO instruction (bidirectional)
                            attention_from_target.append(attn[head_idx, tgt_idx, inst_idx])

                layer_scores[f"layer_{layer_idx}"] = {
                    "instruction_to_target_mean": float(np.mean(attention_to_target)) if attention_to_target else 0.0,
                    "instruction_to_target_max": float(np.max(attention_to_target)) if attention_to_target else 0.0,
                    "target_to_instruction_mean": float(np.mean(attention_from_target)) if attention_from_target else 0.0,
                    "target_to_instruction_max": float(np.max(attention_from_target)) if attention_from_target else 0.0,
                }

                # Store full attention pattern for visualization
                layer_attention_patterns[f"layer_{layer_idx}"] = attn

            results[prompt] = {
                "tokens": tokens,
                "instruction_indices": instruction_indices,
                "target_indices": target_indices,
                "layer_scores": layer_scores,
                "attention_patterns": layer_attention_patterns,
                "hidden_states": activations["hidden_states"]
            }

        return results


if __name__ == "__main__":
    # Quick test
    extractor = AttentionExtractor()

    print("\nModel Info:")
    info = extractor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nTesting with sample prompt...")
    prompt = "Once upon a time, there was a little girl"
    activations = extractor.extract_activations(prompt)

    print(f"Tokens: {activations['tokens']}")
    print(f"Number of layers: {len(activations['attention_weights'])}")
    print(f"Attention shape (layer 0): {activations['attention_weights'][0].shape}")
