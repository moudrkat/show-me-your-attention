"""
Visualization tools for attention patterns and neuron activations.
Creates interactive and static visualizations of model internals.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AttentionVisualizer:
    """
    Creates visualizations of attention patterns and activations.
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            # Fallback if style not available
            pass

    @staticmethod
    def clean_token(token: str) -> str:
        """
        Clean token by removing GPT-2 tokenization artifacts.

        Args:
            token: Raw token string

        Returns:
            Cleaned token string
        """
        # Remove Ġ which represents space in GPT-2 tokenization
        return token.replace('Ġ', ' ').strip()

        self.colors = sns.color_palette("husl", 12)

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        prompt: Optional[str] = None
    ):
        """
        Plot attention weights as a heatmap.

        Args:
            attention_weights: Attention tensor [num_heads, seq_len, seq_len]
            tokens: List of token strings
            layer_idx: Layer index for title
            head_idx: Which attention head to visualize
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
            prompt: Optional prompt text to include in title
        """
        # Extract specific head
        if len(attention_weights.shape) == 4:
            # [batch, heads, seq, seq]
            attn = attention_weights[0, head_idx].cpu().numpy()
        else:
            # [heads, seq, seq]
            attn = attention_weights[head_idx]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=ax,
            cbar_kws={"label": "Attention Weight"}
        )

        title = f"Attention Pattern - Layer {layer_idx}, Head {head_idx}"
        if prompt:
            title += f"\nPrompt: {prompt}"
        ax.set_title(title)
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_all_heads(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        layer_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
        prompt: Optional[str] = None
    ):
        """
        Plot all attention heads for a layer in a grid.

        Args:
            attention_weights: Attention tensor [num_heads, seq_len, seq_len]
            tokens: List of token strings
            layer_idx: Layer index for title
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
            prompt: Optional prompt text to include in title
        """
        if len(attention_weights.shape) == 4:
            attn = attention_weights[0].cpu().numpy()
        else:
            attn = attention_weights

        num_heads = attn.shape[0]
        cols = 4
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            sns.heatmap(
                attn[head_idx],
                ax=ax,
                cmap="viridis",
                cbar=True,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(f"Head {head_idx}")

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis("off")

        title = f"All Attention Heads - Layer {layer_idx}"
        if prompt:
            title += f"\nPrompt: {prompt}"
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_neuron_activations(
        self,
        activations: np.ndarray,
        tokens: List[str],
        layer_idx: int = 0,
        top_k: int = 50,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        prompt: Optional[str] = None
    ):
        """
        Plot neuron activation patterns across tokens.

        Args:
            activations: Hidden state tensor [seq_len, hidden_size]
            tokens: List of token strings
            layer_idx: Layer index for title
            top_k: Number of most active neurons to show
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
            prompt: Optional prompt text to include in title
        """
        # Find top-k most active neurons (by variance)
        neuron_variance = np.var(activations, axis=0)
        top_neuron_indices = np.argsort(neuron_variance)[-top_k:]

        # Extract top neurons
        top_activations = activations[:, top_neuron_indices]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            top_activations.T,
            xticklabels=tokens,
            yticklabels=[f"N{i}" for i in top_neuron_indices],
            cmap="RdBu_r",
            center=0,
            ax=ax,
            cbar_kws={"label": "Activation"}
        )

        title = f"Top {top_k} Neuron Activations - Layer {layer_idx}"
        if prompt:
            title += f"\nPrompt: {prompt}"
        ax.set_title(title)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Neuron Index")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_activation_comparison(
        self,
        activations_dict: Dict[str, np.ndarray],
        layer_idx: int = 0,
        neuron_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Compare neuron activations across different prompts.

        Args:
            activations_dict: Dictionary mapping prompt to activations [seq_len, hidden_size]
            layer_idx: Layer index for title
            neuron_indices: Specific neurons to compare (optional)
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        if neuron_indices is None:
            # Select neurons with highest variance across prompts
            all_activations = np.concatenate([act for act in activations_dict.values()], axis=0)
            neuron_variance = np.var(all_activations, axis=0)
            neuron_indices = np.argsort(neuron_variance)[-20:]

        num_neurons = len(neuron_indices)
        num_prompts = len(activations_dict)

        fig, axes = plt.subplots(1, num_neurons, figsize=figsize, sharey=True)
        if num_neurons == 1:
            axes = [axes]

        for idx, neuron_idx in enumerate(neuron_indices):
            ax = axes[idx]

            for prompt_idx, (prompt, activations) in enumerate(activations_dict.items()):
                neuron_activation = activations[:, neuron_idx]
                ax.plot(
                    neuron_activation,
                    label=prompt[:20] + "...",
                    marker="o",
                    linewidth=2,
                    color=self.colors[prompt_idx % len(self.colors)]
                )

            ax.set_title(f"Neuron {neuron_idx}")
            ax.set_xlabel("Token Position")
            if idx == 0:
                ax.set_ylabel("Activation")
            ax.grid(True, alpha=0.3)

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.suptitle(f"Neuron Activation Comparison - Layer {layer_idx}", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_interactive_attention(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        layer_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Create interactive attention visualization using Plotly.

        Args:
            attention_weights: Attention tensor [num_heads, seq_len, seq_len]
            tokens: List of token strings
            layer_idx: Layer index for title
            save_path: Path to save HTML file (optional)
        """
        if len(attention_weights.shape) == 4:
            attn = attention_weights[0].cpu().numpy()
        else:
            attn = attention_weights

        num_heads = attn.shape[0]

        # Create subplot for each head
        fig = make_subplots(
            rows=(num_heads + 3) // 4,
            cols=4,
            subplot_titles=[f"Head {i}" for i in range(num_heads)]
        )

        for head_idx in range(num_heads):
            row = (head_idx // 4) + 1
            col = (head_idx % 4) + 1

            heatmap = go.Heatmap(
                z=attn[head_idx],
                x=tokens,
                y=tokens,
                colorscale="Viridis",
                showscale=(head_idx == 0)
            )

            fig.add_trace(heatmap, row=row, col=col)

        fig.update_layout(
            title_text=f"Interactive Attention Patterns - Layer {layer_idx}",
            height=300 * ((num_heads + 3) // 4),
            showlegend=False
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive visualization to {save_path}")

        fig.show()

    def plot_layer_statistics(
        self,
        stats_dict: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Plot statistical summary across layers.

        Args:
            stats_dict: Dictionary of layer statistics from get_attention_stats
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        layers = sorted(stats_dict.keys())
        metrics = ["mean", "std", "max", "min"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [stats_dict[layer][metric] for layer in layers]
            ax.plot(range(len(layers)), values, marker="o", linewidth=2)
            ax.set_title(f"Attention {metric.capitalize()} per Layer")
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric.capitalize())
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_instruction_attention_layers(
        self,
        analysis_results: Dict,
        target_word: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        is_causal: bool = True
    ):
        """
        Visualize how target→instruction attention changes across layers for different prompts.

        Args:
            analysis_results: Results from analyze_instruction_attention()
            target_word: The target word being analyzed
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
            is_causal: Whether the model is causal (affects plot titles and interpretation)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # Prepare data
        prompts = list(analysis_results.keys())
        num_layers = len(analysis_results[prompts[0]]["layer_scores"])

        model_type_label = "Causal LM" if is_causal else "Bidirectional LM"

        # Plot 1: Target → Instruction (Mean) - PRIMARY METRIC
        ax = axes[0]
        for prompt_idx, prompt in enumerate(prompts):
            scores = []
            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                score = analysis_results[prompt]["layer_scores"][layer_key]["target_to_instruction_mean"]
                scores.append(score)

            label = prompt[:40] + "..." if len(prompt) > 40 else prompt
            ax.plot(range(num_layers), scores, marker="o", linewidth=2.5,
                   label=label, color=self.colors[prompt_idx % len(self.colors)], markersize=8)

        ax.set_title(f"Target → Instruction (Mean)\nHow '{target_word}' attends to instruction", fontsize=11, weight='bold')
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Mean Attention Weight", fontsize=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Target → Instruction (Max) - STRONGEST HEAD
        ax = axes[1]
        for prompt_idx, prompt in enumerate(prompts):
            scores = []
            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                score = analysis_results[prompt]["layer_scores"][layer_key]["target_to_instruction_max"]
                scores.append(score)

            label = prompt[:40] + "..." if len(prompt) > 40 else prompt
            ax.plot(range(num_layers), scores, marker="s", linewidth=2.5,
                   label=label, color=self.colors[prompt_idx % len(self.colors)], markersize=8)

        ax.set_title(f"Target → Instruction (Max)\nStrongest attention head per layer", fontsize=11, weight='bold')
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Max Attention Weight", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 3: Layer-to-layer change (derivative)
        ax = axes[2]
        for prompt_idx, prompt in enumerate(prompts):
            scores = []
            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                score = analysis_results[prompt]["layer_scores"][layer_key]["target_to_instruction_mean"]
                scores.append(score)

            # Calculate layer-to-layer change
            changes = [0] + [scores[i] - scores[i-1] for i in range(1, len(scores))]

            label = prompt[:40] + "..." if len(prompt) > 40 else prompt
            ax.plot(range(num_layers), changes, marker="^", linewidth=2.5,
                   label=label, color=self.colors[prompt_idx % len(self.colors)], markersize=8)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title("Layer-to-Layer Change\nWhich layers strengthen binding (positive = stronger)", fontsize=11, weight='bold')
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Attention Change", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 4: Attention focus (variance across heads)
        ax = axes[3]
        for prompt_idx, prompt in enumerate(prompts):
            data = analysis_results[prompt]
            focus_scores = []

            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                mean_attn = data["layer_scores"][layer_key]["target_to_instruction_mean"]
                max_attn = data["layer_scores"][layer_key]["target_to_instruction_max"]

                # Focus = max/mean ratio (higher = attention concentrated in fewer heads)
                focus = max_attn / mean_attn if mean_attn > 0 else 0
                focus_scores.append(focus)

            label = prompt[:40] + "..." if len(prompt) > 40 else prompt
            ax.plot(range(num_layers), focus_scores, marker="d", linewidth=2.5,
                   label=label, color=self.colors[prompt_idx % len(self.colors)], markersize=8)

        ax.set_title("Attention Focus (Max/Mean Ratio)\nHigher = more concentrated in few heads", fontsize=11, weight='bold')
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Focus Ratio", fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'Instruction-Target Attention Analysis: "{target_word}" ({model_type_label})',
                    fontsize=14, y=1.00, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_instruction_attention_heatmap(
        self,
        analysis_results: Dict,
        target_word: str,
        metric: str = "instruction_to_target_mean",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Create a heatmap showing instruction→target attention across prompts and layers.

        Args:
            analysis_results: Results from analyze_instruction_attention()
            target_word: The target word being analyzed
            metric: Which metric to visualize (default: "instruction_to_target_mean")
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        prompts = list(analysis_results.keys())
        num_layers = len(analysis_results[prompts[0]]["layer_scores"])

        # Build matrix: rows = prompts, columns = layers
        data = []
        for prompt in prompts:
            row = []
            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                score = analysis_results[prompt]["layer_scores"][layer_key][metric]
                row.append(score)
            data.append(row)

        data = np.array(data)

        fig, ax = plt.subplots(figsize=figsize)

        # Truncate prompt labels
        prompt_labels = [p[:50] + "..." if len(p) > 50 else p for p in prompts]

        sns.heatmap(
            data,
            xticklabels=[f"L{i}" for i in range(num_layers)],
            yticklabels=prompt_labels,
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Attention Weight"},
            annot=True,
            fmt=".3f"
        )

        metric_title = metric.replace("_", " ").title()
        ax.set_title(f'{metric_title} for Target Word: "{target_word}"')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Prompt")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_generated_attention_to_target(
        self,
        generation_result: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        show_layer_breakdown: bool = False
    ):
        """
        Visualize attention from each generated token to the target word.

        Args:
            generation_result: Result from generate_with_attention_to_target()
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
            show_layer_breakdown: If True, show individual layer plots
        """
        generated_tokens = generation_result["generated_tokens"]
        attention_scores = generation_result["attention_scores"]
        target_word = generation_result["target_word"]
        prompt = generation_result["prompt"]

        if show_layer_breakdown:
            # Show layer breakdown
            layer_attention_scores = generation_result["layer_attention_scores"]
            num_layers = len(layer_attention_scores)

            # Create subplots: one for overall, rest for layers
            fig, axes = plt.subplots(2, 1, figsize=figsize)

            # Plot 1: Overall average attention
            ax = axes[0]
            x = range(len(generated_tokens))
            bars = ax.bar(x, attention_scores, alpha=0.7, color='steelblue', edgecolor='black')

            # Color bars by attention strength
            max_score = max(attention_scores) if attention_scores else 1
            for i, (bar, score) in enumerate(zip(bars, attention_scores)):
                normalized = score / max_score if max_score > 0 else 0
                bar.set_color(plt.cm.YlOrRd(normalized))

            ax.set_title(f'Attention to "{target_word}" - Average Across All Layers', fontsize=12, weight='bold')
            ax.set_xlabel('Generated Token Position', fontsize=10)
            ax.set_ylabel('Attention Score', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{i}\n{tok}" for i, tok in enumerate(generated_tokens)], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

            # Plot 2: Layer heatmap
            ax = axes[1]
            layer_data = []
            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                layer_data.append(layer_attention_scores[layer_key])

            layer_data = np.array(layer_data)

            im = ax.imshow(layer_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_title('Attention per Layer (Heatmap)', fontsize=12, weight='bold')
            ax.set_xlabel('Generated Token Position', fontsize=10)
            ax.set_ylabel('Layer', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{i}\n{tok}" for i, tok in enumerate(generated_tokens)], rotation=45, ha='right')
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels([f"L{i}" for i in range(num_layers)])

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attention Score', fontsize=10)

            fig.suptitle(f'Generated Tokens Attention to "{target_word}"\nPrompt: {prompt}',
                        fontsize=13, weight='bold')

        else:
            # Simple single plot
            fig, ax = plt.subplots(figsize=figsize)

            x = range(len(generated_tokens))
            bars = ax.bar(x, attention_scores, alpha=0.7, edgecolor='black', linewidth=1.5)

            # Color bars by attention strength (gradient)
            max_score = max(attention_scores) if attention_scores else 1
            for i, (bar, score) in enumerate(zip(bars, attention_scores)):
                normalized = score / max_score if max_score > 0 else 0
                bar.set_color(plt.cm.YlOrRd(normalized))

            ax.set_title(f'Attention from Generated Tokens to "{target_word}"',
                        fontsize=14, weight='bold')
            ax.set_xlabel('Generated Token', fontsize=11)
            ax.set_ylabel('Attention Score (Avg Across Layers)', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{i}\n{tok}" for i, tok in enumerate(generated_tokens)],
                              rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add text annotation for prompt
            fig.text(0.5, 0.95, f'Prompt: "{prompt}"', ha='center', fontsize=10,
                    wrap=True, style='italic')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_generated_attention_to_all_prompt_tokens(
        self,
        generation_result: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        show_probabilities: bool = True
    ):
        """
        Visualize attention from generated tokens to each prompt token.
        Creates one subplot per prompt token (stacked vertically), showing how all generated tokens attend to it.

        Args:
            generation_result: Result from generate_with_attention_to_all_prompt_tokens()
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
            show_probabilities: Whether to display token probabilities on the x-axis labels
        """
        generated_tokens = generation_result["generated_tokens"]
        prompt_tokens = generation_result["prompt_tokens"]
        attention_to_prompt = generation_result["attention_to_prompt"]  # List of [num_prompt_tokens] arrays
        prompt = generation_result["prompt"]
        token_probabilities = generation_result.get("token_probabilities", None)

        num_prompt_tokens = len(prompt_tokens)
        num_generated = len(generated_tokens)

        # Create attention matrix: rows = generated tokens, cols = prompt tokens
        attention_matrix = np.array(attention_to_prompt)  # Shape: [num_generated, num_prompt_tokens]

        # Create subplots: one per prompt token, stacked vertically
        fig, axes = plt.subplots(num_prompt_tokens, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Plot attention to each prompt token
        for prompt_idx, prompt_token in enumerate(prompt_tokens):
            ax = axes[prompt_idx]

            # Get attention from all generated tokens to this prompt token
            attention_scores = attention_matrix[:, prompt_idx]

            x = range(num_generated)
            bars = ax.bar(x, attention_scores, alpha=0.7, edgecolor='black', linewidth=1)

            # Color bars by attention strength
            max_score = max(attention_scores) if len(attention_scores) > 0 and max(attention_scores) > 0 else 1
            for i, (bar, score) in enumerate(zip(bars, attention_scores)):
                normalized = score / max_score if max_score > 0 else 0
                bar.set_color(plt.cm.viridis(normalized))

            # Clean token display (remove Ġ which represents space in GPT-2 tokenization)
            clean_token = prompt_token.replace('Ġ', ' ').strip()
            ax.set_title(f'Prompt Token: "{clean_token}"', fontsize=11, weight='bold', loc='left')
            ax.set_ylabel('Attention', fontsize=9)
            ax.set_xticks(x)

            # Create x-tick labels with optional probabilities
            if show_probabilities and token_probabilities is not None:
                xticklabels = [f"{i}\n{tok}\np={token_probabilities[i]:.3f}"
                              for i, tok in enumerate(generated_tokens)]
            else:
                xticklabels = [f"{i}\n{tok}" for i, tok in enumerate(generated_tokens)]

            ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=7)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='y', labelsize=8)

            # Only show x-label on bottom plot
            if prompt_idx == num_prompt_tokens - 1:
                xlabel = 'Generated Token (Index + Text + Probability)' if show_probabilities and token_probabilities else 'Generated Token (Index + Text)'
                ax.set_xlabel(xlabel, fontsize=9)

        fig.suptitle(f'Attention from Generated Tokens to Each Prompt Token\nPrompt: "{prompt}"',
                    fontsize=13, weight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()

    def plot_attention_vs_embedding_similarity(
        self,
        analysis_results: Dict,
        target_word: str,
        instruction_word: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Compare attention scores with embedding cosine similarity across layers.
        This shows whether high attention correlates with similar embeddings.

        Args:
            analysis_results: Dictionary with analysis results containing hidden_states
            target_word: The target word to track
            instruction_word: The instruction word to track
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        from sklearn.metrics.pairwise import cosine_similarity

        prompts = list(analysis_results.keys())
        num_prompts = len(prompts)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # Collect data for correlation plot
        all_attention = []
        all_similarity = []
        all_prompts_labels = []

        for prompt_idx, (prompt, data) in enumerate(analysis_results.items()):
            target_indices = data.get('target_indices', [])
            instruction_indices = data.get('instruction_indices', [])

            if not target_indices or not instruction_indices or 'hidden_states' not in data:
                continue

            hidden_states = data['hidden_states']
            layer_scores = data.get('layer_scores', {})

            # Use the minimum of hidden_states length and layer_scores length
            num_hidden_layers = len(hidden_states)
            num_score_layers = len(layer_scores)
            num_layers = min(num_hidden_layers, num_score_layers)

            if num_layers == 0:
                continue

            # Calculate cosine similarity across layers
            cosine_similarities = []
            attention_scores = []

            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"

                # Skip if this layer doesn't have scores
                if layer_key not in layer_scores:
                    continue

                layer_hidden = hidden_states[layer_idx].squeeze(0).cpu().numpy()

                # Average embeddings for target and instruction words
                target_emb = np.mean([layer_hidden[idx] for idx in target_indices], axis=0)
                instruction_emb = np.mean([layer_hidden[idx] for idx in instruction_indices], axis=0)

                # Compute cosine similarity
                cos_sim = cosine_similarity(
                    target_emb.reshape(1, -1),
                    instruction_emb.reshape(1, -1)
                )[0, 0]
                cosine_similarities.append(cos_sim)

                # Get attention score for this layer
                attn_score = layer_scores[layer_key]['target_to_instruction_mean']
                attention_scores.append(attn_score)

                # Collect for correlation plot
                all_attention.append(attn_score)
                all_similarity.append(cos_sim)
                all_prompts_labels.append(prompt_idx)

            # Plot 1: Cosine Similarity across layers
            ax = axes[0]
            label = prompt[:30] + "..." if len(prompt) > 30 else prompt
            color = self.colors[prompt_idx % len(self.colors)]
            ax.plot(range(num_layers), cosine_similarities, marker='o', linewidth=2.5,
                   label=label, color=color, markersize=8)

            # Plot 2: Attention scores across layers (for comparison)
            ax = axes[1]
            ax.plot(range(num_layers), attention_scores, marker='s', linewidth=2.5,
                   label=label, color=color, markersize=8)

        # Configure Plot 1
        axes[0].set_title('Cosine Similarity: Target ↔ Instruction Embeddings',
                         fontsize=12, weight='bold')
        axes[0].set_xlabel('Layer', fontsize=10)
        axes[0].set_ylabel('Cosine Similarity', fontsize=10)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Configure Plot 2
        axes[1].set_title('Attention Score: Target → Instruction (Mean)',
                         fontsize=12, weight='bold')
        axes[1].set_xlabel('Layer', fontsize=10)
        axes[1].set_ylabel('Attention Score', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Correlation scatter plot (all prompts combined)
        ax = axes[2]
        for prompt_idx in range(num_prompts):
            mask = [p == prompt_idx for p in all_prompts_labels]
            attn = [all_attention[i] for i, m in enumerate(mask) if m]
            sim = [all_similarity[i] for i, m in enumerate(mask) if m]

            color = self.colors[prompt_idx % len(self.colors)]
            label = prompts[prompt_idx][:30] + "..." if len(prompts[prompt_idx]) > 30 else prompts[prompt_idx]

            ax.scatter(attn, sim, c=[color], label=label, alpha=0.7, s=100,
                      edgecolors='black', linewidth=0.5)

        # Add correlation coefficient
        if len(all_attention) > 1:
            correlation = np.corrcoef(all_attention, all_similarity)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=11, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top')

        ax.set_title('Correlation: Attention vs Embedding Similarity',
                    fontsize=12, weight='bold')
        ax.set_xlabel('Attention Score (Target → Instruction)', fontsize=10)
        ax.set_ylabel('Cosine Similarity (Embeddings)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 4: Difference (Attention - Normalized Similarity) across layers
        ax = axes[3]
        for prompt_idx, (prompt, data) in enumerate(analysis_results.items()):
            target_indices = data.get('target_indices', [])
            instruction_indices = data.get('instruction_indices', [])

            if not target_indices or not instruction_indices:
                continue

            hidden_states = data['hidden_states']
            layer_scores = data.get('layer_scores', {})

            # Use the minimum of hidden_states length and layer_scores length
            num_hidden_layers = len(hidden_states)
            num_score_layers = len(layer_scores)
            num_layers = min(num_hidden_layers, num_score_layers)

            if num_layers == 0:
                continue

            cosine_similarities = []
            attention_scores = []

            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"

                # Skip if this layer doesn't have scores
                if layer_key not in layer_scores:
                    continue

                layer_hidden = hidden_states[layer_idx].squeeze(0).cpu().numpy()

                target_emb = np.mean([layer_hidden[idx] for idx in target_indices], axis=0)
                instruction_emb = np.mean([layer_hidden[idx] for idx in instruction_indices], axis=0)

                cos_sim = cosine_similarity(
                    target_emb.reshape(1, -1),
                    instruction_emb.reshape(1, -1)
                )[0, 0]
                cosine_similarities.append(cos_sim)

                attn_score = layer_scores[layer_key]['target_to_instruction_mean']
                attention_scores.append(attn_score)

            # Normalize both to [0, 1] for comparison
            if len(attention_scores) > 0:
                attn_norm = np.array(attention_scores)
                sim_norm = (np.array(cosine_similarities) + 1) / 2  # Cosine is in [-1, 1]

                difference = attn_norm - sim_norm

                label = prompt[:30] + "..." if len(prompt) > 30 else prompt
                color = self.colors[prompt_idx % len(self.colors)]
                ax.plot(range(num_layers), difference, marker='d', linewidth=2.5,
                       label=label, color=color, markersize=8)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_title('Divergence: Attention - Similarity (normalized)',
                    fontsize=12, weight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Attention - Similarity', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'Positive = More attention than similarity\nNegative = More similarity than attention',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        fig.suptitle(f'Attention vs Embedding Similarity: "{target_word}" ↔ "{instruction_word}"',
                    fontsize=14, y=0.995, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def compute_attention_entropy(
        self,
        attention_weights: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compute Shannon entropy of attention distribution.

        Higher entropy = more diffuse attention across tokens
        Lower entropy = more focused attention on specific tokens

        Args:
            attention_weights: Attention weights [seq_len] or [heads, seq_len]
            epsilon: Small value to avoid log(0)

        Returns:
            Entropy value (in nats if using natural log)
        """
        # Ensure positive and normalized
        attn = np.array(attention_weights)
        if len(attn.shape) > 1:
            # Average across heads if multi-head
            attn = np.mean(attn, axis=0)

        # Normalize to probability distribution
        attn = attn + epsilon
        attn = attn / np.sum(attn)

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(attn * np.log(attn + epsilon))

        return float(entropy)

    def compute_cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """
        Compute Cohen's d effect size between two groups.

        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size
        """
        group1 = np.array(group1)
        group2 = np.array(group2)

        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0

        return float(d)

    def plot_attention_entropy_analysis(
        self,
        analysis_results: Dict,
        target_word: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        Visualize attention entropy across layers for different prompts.

        Shows how focused vs diffuse the attention patterns are.

        Args:
            analysis_results: Results from analyze_instruction_attention()
            target_word: The target word being analyzed
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        prompts = list(analysis_results.keys())
        num_layers = len(analysis_results[prompts[0]]["layer_scores"])

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Entropy across layers
        ax = axes[0]
        for prompt_idx, (prompt, data) in enumerate(analysis_results.items()):
            entropies = []

            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                if 'attention_patterns' in data and layer_idx < len(data['attention_patterns']):
                    # Get attention pattern for this layer
                    attn_pattern = data['attention_patterns'][layer_idx]

                    # Compute average entropy across target tokens
                    target_indices = data.get('target_indices', [])
                    if target_indices and len(attn_pattern.shape) >= 2:
                        layer_entropies = []
                        for target_idx in target_indices:
                            if target_idx < attn_pattern.shape[-2]:
                                # Get attention from this target token
                                target_attn = attn_pattern[:, target_idx, :] if len(attn_pattern.shape) == 3 else attn_pattern[target_idx, :]
                                entropy = self.compute_attention_entropy(target_attn)
                                layer_entropies.append(entropy)
                        entropies.append(np.mean(layer_entropies) if layer_entropies else 0)
                    else:
                        entropies.append(0)
                else:
                    entropies.append(0)

            label = prompt[:40] + "..." if len(prompt) > 40 else prompt
            color = self.colors[prompt_idx % len(self.colors)]
            ax.plot(range(num_layers), entropies, marker='o', linewidth=2.5,
                   label=label, color=color, markersize=8)

        ax.set_title(f'Attention Entropy: Target "{target_word}"\nHigher = more diffuse attention',
                    fontsize=12, weight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Entropy (nats)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 2: Average entropy comparison
        ax = axes[1]
        avg_entropies = []
        prompt_labels = []

        for prompt_idx, (prompt, data) in enumerate(analysis_results.items()):
            entropies = []
            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                if 'attention_patterns' in data and layer_idx < len(data['attention_patterns']):
                    attn_pattern = data['attention_patterns'][layer_idx]
                    target_indices = data.get('target_indices', [])
                    if target_indices and len(attn_pattern.shape) >= 2:
                        layer_entropies = []
                        for target_idx in target_indices:
                            if target_idx < attn_pattern.shape[-2]:
                                target_attn = attn_pattern[:, target_idx, :] if len(attn_pattern.shape) == 3 else attn_pattern[target_idx, :]
                                entropy = self.compute_attention_entropy(target_attn)
                                layer_entropies.append(entropy)
                        entropies.append(np.mean(layer_entropies) if layer_entropies else 0)

            if entropies:
                avg_entropies.append(np.mean(entropies))
                prompt_labels.append(prompt[:30] + "..." if len(prompt) > 30 else prompt)

        if avg_entropies:
            bars = ax.bar(range(len(avg_entropies)), avg_entropies,
                         color=[self.colors[i % len(self.colors)] for i in range(len(avg_entropies))],
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_title('Average Attention Entropy\n(across all layers)', fontsize=12, weight='bold')
            ax.set_xlabel('Prompt', fontsize=10)
            ax.set_ylabel('Mean Entropy', fontsize=10)
            ax.set_xticks(range(len(prompt_labels)))
            ax.set_xticklabels(prompt_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_effect_size_heatmap(
        self,
        analysis_results: Dict,
        target_word: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Create heatmap of Cohen's d effect sizes between all prompt pairs.

        Shows magnitude of attention differences between prompts.

        Args:
            analysis_results: Results from analyze_instruction_attention()
            target_word: The target word being analyzed
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        prompts = list(analysis_results.keys())
        num_prompts = len(prompts)
        num_layers = len(analysis_results[prompts[0]]["layer_scores"])

        # Create subplots: one for each layer showing prompt pair comparisons
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Compute effect sizes for each layer
        layer_effect_sizes = np.zeros((num_layers, num_prompts, num_prompts))

        for layer_idx in range(num_layers):
            layer_key = f"layer_{layer_idx}"

            # Collect attention scores for this layer
            layer_scores = []
            for prompt in prompts:
                score = analysis_results[prompt]["layer_scores"][layer_key]["target_to_instruction_mean"]
                layer_scores.append(score)

            # Compute pairwise effect sizes
            for i in range(num_prompts):
                for j in range(num_prompts):
                    if i != j:
                        # For effect size, treat each layer's score as a "sample"
                        # We'll use all layers as samples for each prompt
                        prompt_i_scores = [
                            analysis_results[prompts[i]]["layer_scores"][f"layer_{l}"]["target_to_instruction_mean"]
                            for l in range(num_layers)
                        ]
                        prompt_j_scores = [
                            analysis_results[prompts[j]]["layer_scores"][f"layer_{l}"]["target_to_instruction_mean"]
                            for l in range(num_layers)
                        ]

                        d = self.compute_cohens_d(prompt_i_scores, prompt_j_scores)
                        layer_effect_sizes[layer_idx, i, j] = d

        # Plot 1: Average effect size across all layers
        ax = axes[0]
        avg_effect_sizes = np.mean(layer_effect_sizes, axis=0)

        prompt_labels = [p[:25] + "..." if len(p) > 25 else p for p in prompts]

        im = ax.imshow(avg_effect_sizes, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax.set_xticks(range(num_prompts))
        ax.set_yticks(range(num_prompts))
        ax.set_xticklabels(prompt_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(prompt_labels, fontsize=9)

        # Add text annotations
        for i in range(num_prompts):
            for j in range(num_prompts):
                if i != j:
                    text = ax.text(j, i, f'{avg_effect_sizes[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)

        ax.set_title("Cohen's d Effect Sizes\n(Average across layers)", fontsize=12, weight='bold')
        ax.set_xlabel('Prompt (compared to)', fontsize=10)
        ax.set_ylabel('Prompt', fontsize=10)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Effect Size (Cohen's d)", fontsize=10)

        # Plot 2: Effect size interpretation guide + max effects
        ax = axes[1]
        ax.axis('off')

        # Find largest effect sizes
        abs_effects = np.abs(avg_effect_sizes)
        np.fill_diagonal(abs_effects, 0)  # Ignore diagonal
        max_indices = np.unravel_index(np.argsort(abs_effects.ravel())[-3:], abs_effects.shape)

        text_content = "Effect Size Interpretation:\n\n"
        text_content += "|d| < 0.2: Negligible\n"
        text_content += "0.2 ≤ |d| < 0.5: Small\n"
        text_content += "0.5 ≤ |d| < 0.8: Medium\n"
        text_content += "|d| ≥ 0.8: Large\n\n"
        text_content += "─" * 40 + "\n\n"
        text_content += "Largest Effects:\n\n"

        for idx in range(len(max_indices[0])):
            i, j = max_indices[0][idx], max_indices[1][idx]
            d_val = avg_effect_sizes[i, j]
            magnitude = "Large" if abs(d_val) >= 0.8 else "Medium" if abs(d_val) >= 0.5 else "Small" if abs(d_val) >= 0.2 else "Negligible"
            direction = "higher" if d_val > 0 else "lower"

            text_content += f"{idx+1}. {prompts[i][:20]}...\n"
            text_content += f"   vs {prompts[j][:20]}...\n"
            text_content += f"   d = {d_val:.3f} ({magnitude})\n"
            text_content += f"   → Prompt 1 has {direction} attention\n\n"

        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        fig.suptitle(f'Effect Size Analysis: Target "{target_word}"',
                    fontsize=14, y=0.98, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()


    def plot_qkv_matrices(
        self,
        qkv_data: Dict,
        head_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (24, 12)
    ):
        """
        Visualize Query, Key, Value matrices as heatmaps for a specific attention head.
        Shows the complete attention calculation flow with visual separators.

        Args:
            qkv_data: Dictionary from extract_qkv_matrices() containing Q, K, V matrices
            head_idx: Which attention head to visualize (0 to num_heads-1)
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        tokens = qkv_data["tokens"]
        Q = qkv_data["Q"][head_idx]  # [seq_len, head_dim]
        K = qkv_data["K"][head_idx]  # [seq_len, head_dim]
        V = qkv_data["V"][head_idx]  # [seq_len, head_dim]
        layer_idx = qkv_data["layer_idx"]
        num_heads = qkv_data["num_heads"]
        head_dim = qkv_data["head_dim"]

        # Create figure with custom layout to include multiplication symbols
        fig = plt.figure(figsize=figsize)

        # Use GridSpec for better control over spacing
        import matplotlib.gridspec as gridspec
        # Row 1: Q × K^T → Scores → Softmax → Attention
        # Row 2: Attention × V → Output
        gs = gridspec.GridSpec(4, 7, figure=fig,
                              width_ratios=[1, 0.15, 1, 0.15, 1, 0.15, 1],
                              height_ratios=[1, 0.15, 0.8, 0.15],
                              hspace=0.3, wspace=0.4,
                              top=0.88)

        # Top row: Q, K, V matrices with multiplication symbols
        # Plot Q matrix
        ax_q = fig.add_subplot(gs[0, 0])
        im_q = ax_q.imshow(Q, aspect='auto', cmap='Blues', interpolation='nearest')
        ax_q.set_title('Query (Q)\n"What to look for"', fontsize=11, weight='bold', color='darkblue')
        ax_q.set_xlabel(f'Dimension (d={head_dim})', fontsize=9)
        ax_q.set_ylabel('Token', fontsize=9)
        ax_q.set_yticks(range(len(tokens)))
        ax_q.set_yticklabels([self.clean_token(t) for t in tokens], fontsize=8)
        # Add shape annotation
        ax_q.text(0.5, -0.15, f'Shape: [{len(tokens)} × {head_dim}]',
                 transform=ax_q.transAxes, ha='center', fontsize=8,
                 style='italic', color='darkblue')

        # Multiplication symbol between Q and K
        ax_mult1 = fig.add_subplot(gs[0, 1])
        ax_mult1.axis('off')
        ax_mult1.text(0.5, 0.5, '×', fontsize=40, ha='center', va='center',
                     weight='bold', color='darkred')
        ax_mult1.text(0.5, 0.2, 'matmul', fontsize=8, ha='center', va='center',
                     style='italic', color='gray')

        # Plot K matrix (transposed for visualization)
        ax_k = fig.add_subplot(gs[0, 2])
        im_k = ax_k.imshow(K.T, aspect='auto', cmap='Greens', interpolation='nearest')
        ax_k.set_title('Key (K)ᵀ\n"What I contain"', fontsize=11, weight='bold', color='darkgreen')
        ax_k.set_xlabel('Token', fontsize=9)
        ax_k.set_ylabel(f'Dimension (d={head_dim})', fontsize=9)
        ax_k.set_xticks(range(len(tokens)))
        ax_k.set_xticklabels([self.clean_token(t) for t in tokens], fontsize=8, rotation=45, ha='right')
        # Add shape annotation
        ax_k.text(0.5, -0.25, f'Shape: [{head_dim} × {len(tokens)}]',
                 transform=ax_k.transAxes, ha='center', fontsize=8,
                 style='italic', color='darkgreen')

        # Arrow/equals symbol
        ax_arrow1 = fig.add_subplot(gs[0, 3])
        ax_arrow1.axis('off')
        ax_arrow1.text(0.5, 0.5, '→', fontsize=40, ha='center', va='center',
                      weight='bold', color='purple')
        ax_arrow1.text(0.5, 0.2, 'produces', fontsize=8, ha='center', va='center',
                      style='italic', color='gray')

        # Compute and plot attention scores (Q @ K^T / sqrt(d_k))
        scores = np.matmul(Q, K.T) / np.sqrt(head_dim)
        ax_scores = fig.add_subplot(gs[0, 4])
        im_scores = ax_scores.imshow(scores, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        ax_scores.set_title('Scores\nQ·Kᵀ/√d', fontsize=11, weight='bold', color='purple')
        ax_scores.set_xlabel('Key Token', fontsize=9)
        ax_scores.set_ylabel('Query Token', fontsize=9)
        ax_scores.set_xticks(range(len(tokens)))
        ax_scores.set_yticks(range(len(tokens)))
        ax_scores.set_xticklabels([self.clean_token(t) for t in tokens], fontsize=7, rotation=45, ha='right')
        ax_scores.set_yticklabels([self.clean_token(t) for t in tokens], fontsize=7)
        # Add shape annotation
        ax_scores.text(0.5, -0.25, f'Shape: [{len(tokens)} × {len(tokens)}]',
                      transform=ax_scores.transAxes, ha='center', fontsize=8,
                      style='italic', color='purple')

        # Softmax arrow
        ax_arrow2 = fig.add_subplot(gs[0, 5])
        ax_arrow2.axis('off')
        ax_arrow2.text(0.5, 0.5, '→', fontsize=40, ha='center', va='center',
                      weight='bold', color='orange')
        ax_arrow2.text(0.5, 0.2, 'softmax', fontsize=8, ha='center', va='center',
                      style='italic', color='gray')

        # Compute and plot attention weights (after softmax)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        ax_attn = fig.add_subplot(gs[0, 6])
        im_attn = ax_attn.imshow(attention, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax_attn.set_title('Attention\nWeights', fontsize=11, weight='bold', color='darkorange')
        ax_attn.set_xlabel('Key Token', fontsize=9)
        ax_attn.set_ylabel('Query Token', fontsize=9)
        ax_attn.set_xticks(range(len(tokens)))
        ax_attn.set_yticks(range(len(tokens)))
        ax_attn.set_xticklabels([self.clean_token(t) for t in tokens], fontsize=7, rotation=45, ha='right')
        ax_attn.set_yticklabels([self.clean_token(t) for t in tokens], fontsize=7)
        # Add shape annotation
        ax_attn.text(0.5, -0.30, f'Shape: [{len(tokens)} × {len(tokens)}]',
                    transform=ax_attn.transAxes, ha='center', fontsize=8,
                    style='italic', color='darkorange')

        # Second row: Show Attention × V → Output
        # Plot Attention weights (same as above, for the multiplication)
        ax_attn2 = fig.add_subplot(gs[2, 0])
        im_attn2 = ax_attn2.imshow(attention, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax_attn2.set_title('Attention\nWeights', fontsize=11, weight='bold', color='darkorange')
        ax_attn2.set_xlabel('Key Token', fontsize=9)
        ax_attn2.set_ylabel('Query Token', fontsize=9)
        ax_attn2.set_xticks(range(len(tokens)))
        ax_attn2.set_yticks(range(len(tokens)))
        ax_attn2.set_xticklabels([self.clean_token(t) for t in tokens], fontsize=7, rotation=45, ha='right')
        ax_attn2.set_yticklabels([self.clean_token(t) for t in tokens], fontsize=7)
        ax_attn2.text(0.5, -0.2, f'[{len(tokens)} × {len(tokens)}]',
                     transform=ax_attn2.transAxes, ha='center', fontsize=8,
                     style='italic', color='darkorange')
        fig.colorbar(im_attn2, ax=ax_attn2, fraction=0.046, pad=0.04)

        # Multiplication symbol between Attention and V
        ax_mult2 = fig.add_subplot(gs[2, 1])
        ax_mult2.axis('off')
        ax_mult2.text(0.5, 0.5, '×', fontsize=40, ha='center', va='center',
                     weight='bold', color='darkred')
        ax_mult2.text(0.5, 0.2, 'matmul', fontsize=8, ha='center', va='center',
                     style='italic', color='gray')

        # Plot V matrix
        ax_v = fig.add_subplot(gs[2, 2])
        im_v = ax_v.imshow(V, aspect='auto', cmap='Purples', interpolation='nearest')
        ax_v.set_title('Value (V)\n"What to carry"', fontsize=11, weight='bold', color='purple')
        ax_v.set_xlabel(f'Dimension (d={head_dim})', fontsize=9)
        ax_v.set_ylabel('Token', fontsize=9)
        ax_v.set_yticks(range(len(tokens)))
        ax_v.set_yticklabels([self.clean_token(t) for t in tokens], fontsize=8)
        ax_v.text(0.5, -0.2, f'[{len(tokens)} × {head_dim}]',
                 transform=ax_v.transAxes, ha='center', fontsize=8,
                 style='italic', color='purple')
        fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)

        # Equals/Arrow symbol
        ax_arrow3 = fig.add_subplot(gs[2, 3])
        ax_arrow3.axis('off')
        ax_arrow3.text(0.5, 0.5, '=', fontsize=40, ha='center', va='center',
                      weight='bold', color='darkgreen')
        ax_arrow3.text(0.5, 0.2, 'produces', fontsize=8, ha='center', va='center',
                      style='italic', color='gray')

        # Compute and plot output (Attention @ V)
        output = np.matmul(attention, V)
        ax_output = fig.add_subplot(gs[2, 4])
        im_output = ax_output.imshow(output, aspect='auto', cmap='viridis', interpolation='nearest')
        ax_output.set_title('Output\n(Attention @ V)', fontsize=11, weight='bold', color='darkgreen')
        ax_output.set_xlabel(f'Dimension (d={head_dim})', fontsize=9)
        ax_output.set_ylabel('Token', fontsize=9)
        ax_output.set_yticks(range(len(tokens)))
        ax_output.set_yticklabels([self.clean_token(t) for t in tokens], fontsize=8)
        ax_output.text(0.5, -0.2, f'[{len(tokens)} × {head_dim}]',
                      transform=ax_output.transAxes, ha='center', fontsize=8,
                      style='italic', color='darkgreen')
        fig.colorbar(im_output, ax=ax_output, fraction=0.046, pad=0.04)

        # Bottom row: Summary text
        ax_summary = fig.add_subplot(gs[3, :])
        ax_summary.axis('off')
        summary_text = (
            'Complete Attention Flow: Q·K^T/√d → softmax → Attention Weights → Attention·V → Final Output'
        )
        ax_summary.text(0.5, 0.5, summary_text, fontsize=11, ha='center', va='center',
                       weight='bold', family='monospace',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue',
                                edgecolor='navy', linewidth=2))

        # Add remaining colorbars (top row)
        fig.colorbar(im_q, ax=ax_q, fraction=0.046, pad=0.04)
        fig.colorbar(im_k, ax=ax_k, fraction=0.046, pad=0.04)
        fig.colorbar(im_scores, ax=ax_scores, fraction=0.046, pad=0.04)
        fig.colorbar(im_attn, ax=ax_attn, fraction=0.046, pad=0.04)

        # Main title
        fig.suptitle(f'Attention Mechanism: Complete Q, K, V Computation Flow\n'
                    f'Layer {layer_idx}, Head {head_idx}/{num_heads}\n'
                    f'Row 1: Q × K^T → Scores → Softmax → Attention | Row 2: Attention × V → Output',
                    fontsize=13, y=0.97, weight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_qkv_all_heads(
        self,
        qkv_data: Dict,
        matrix_type: str = "Q",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 12)
    ):
        """
        Visualize one type of matrix (Q, K, or V) across all attention heads.

        Args:
            qkv_data: Dictionary from extract_qkv_matrices() containing Q, K, V matrices
            matrix_type: Which matrix to visualize ("Q", "K", or "V")
            save_path: Path to save figure (optional)
            figsize: Figure size tuple
        """
        tokens = qkv_data["tokens"]
        layer_idx = qkv_data["layer_idx"]
        num_heads = qkv_data["num_heads"]

        if matrix_type == "Q":
            matrices = qkv_data["Q"]  # [num_heads, seq_len, head_dim]
            title = "Query (Q)"
        elif matrix_type == "K":
            matrices = qkv_data["K"]
            title = "Key (K)"
        elif matrix_type == "V":
            matrices = qkv_data["V"]
            title = "Value (V)"
        else:
            raise ValueError(f"matrix_type must be 'Q', 'K', or 'V', got {matrix_type}")

        # Create grid of subplots
        cols = 4
        rows = (num_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]

        # Find global min/max for consistent color scale
        vmin = matrices.min()
        vmax = matrices.max()

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            matrix = matrices[head_idx]  # [seq_len, head_dim]

            im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                          interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.set_xlabel('Dim', fontsize=8)
            ax.set_ylabel('Token', fontsize=8)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels([self.clean_token(t) for t in tokens], fontsize=6)

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        # Add colorbar
        fig.colorbar(im, ax=axes, orientation='horizontal',
                    fraction=0.02, pad=0.04, label='Value')

        fig.suptitle(f'{title} Matrices Across All Heads (Layer {layer_idx})',
                    fontsize=16, y=0.995, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    print("Visualizer module loaded successfully!")
    print("Use AttentionVisualizer class to create visualizations.")
