"""
Streamlit app for visualizing sampling parameters: Temperature, Top-K, and Top-P.
Shows how these parameters affect text generation probability distributions.
"""

import streamlit as st
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_loader import AttentionExtractor

# Set page config
st.set_page_config(
    page_title="LLM Sampling Parameters Visualizer",
    page_icon="🎲",
    layout="wide"
)

# Initialize session state
if 'extractor' not in st.session_state:
    st.session_state.extractor = None

if 'generation_results' not in st.session_state:
    st.session_state.generation_results = {}


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits."""
    if k <= 0:
        return logits

    top_k_logits, top_k_indices = torch.topk(logits, min(k, logits.size(-1)))

    # Set all non-top-k logits to -inf
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, top_k_indices, top_k_logits)

    return filtered_logits


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits."""
    if p >= 1.0:
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Calculate cumulative probabilities
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Find the cutoff index where cumulative probability exceeds p
    sorted_indices_to_remove = cumulative_probs > p

    # Keep at least one token
    sorted_indices_to_remove[0] = False

    # Create filtered logits
    filtered_logits = logits.clone()
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    filtered_logits[indices_to_remove] = float('-inf')

    return filtered_logits


def generate_with_sampling_visualization(
    extractor: AttentionExtractor,
    prompt: str,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_new_tokens: int = 20,
    num_tokens_to_visualize: int = 3
) -> Dict:
    """
    Generate text while capturing probability distributions at each step.

    Returns:
        Dictionary containing:
            - generated_text: Full generated text
            - steps: List of dictionaries, each containing:
                - token: Generated token string
                - token_id: Token ID
                - original_probs: Original probability distribution (top N)
                - temp_probs: After temperature scaling (top N)
                - topk_probs: After top-k filtering (top N)
                - topp_probs: After top-p filtering (top N)
                - final_probs: Final probability distribution (top N)
                - top_tokens: List of (token_string, probability) tuples for top N tokens
    """
    # Tokenize prompt
    inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
    current_ids = inputs["input_ids"]

    steps = []

    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            # Forward pass
            outputs = extractor.model(input_ids=current_ids, return_dict=True)

            # Get logits for the last token
            logits = outputs.logits[0, -1, :].clone()  # [vocab_size]

            # Store original probabilities (for determining top tokens)
            original_probs = torch.softmax(logits, dim=-1)

            # Apply temperature to logits (BEFORE softmax)
            temp_logits = apply_temperature(logits, temperature)
            temp_probs = torch.softmax(temp_logits, dim=-1)

            # Apply top-k filtering to temperature-scaled logits
            topk_logits = apply_top_k(temp_logits.clone(), top_k)
            topk_probs = torch.softmax(topk_logits, dim=-1)

            # Apply top-p filtering
            topp_logits = apply_top_p(topk_logits.clone(), top_p)
            final_probs = torch.softmax(topp_logits, dim=-1)

            # Sample from final distribution
            next_token_id = torch.multinomial(final_probs, num_samples=1)

            # Get top N tokens for visualization
            top_n_probs, top_n_indices = torch.topk(final_probs, k=min(num_tokens_to_visualize, len(final_probs)))
            top_tokens = [
                (extractor.tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top_n_indices, top_n_probs)
            ]

            # Decode the selected token
            token_str = extractor.tokenizer.decode([next_token_id.item()])

            # Get top tokens from original distribution for consistent labeling
            top_original_indices = torch.argsort(original_probs, descending=True)[:num_tokens_to_visualize]
            top_original_tokens = [
                extractor.tokenizer.decode([idx.item()]).replace('Ġ', ' ').strip()
                for idx in top_original_indices
            ]

            # Store step information (including raw logits)
            step_info = {
                'token': token_str,
                'token_id': next_token_id.item(),
                'original_logits': logits.cpu().numpy(),  # Raw logits
                'temp_logits': temp_logits.cpu().numpy(),  # Temperature-scaled logits
                'original_probs': original_probs.cpu().numpy(),
                'temp_probs': temp_probs.cpu().numpy(),
                'topk_probs': topk_probs.cpu().numpy(),
                'topp_probs': final_probs.cpu().numpy(),
                'final_probs': final_probs.cpu().numpy(),
                'top_tokens': top_tokens,
                'selected_token_prob': final_probs[next_token_id].item(),
                'top_token_indices': top_original_indices.cpu().numpy(),
                'top_token_labels': top_original_tokens,
                'tokenizer': extractor.tokenizer
            }
            steps.append(step_info)

            # Append token to sequence
            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)

            # Check for EOS
            if next_token_id.item() == extractor.tokenizer.eos_token_id:
                break

    # Decode full text
    generated_text = extractor.tokenizer.decode(current_ids[0], skip_special_tokens=True)

    return {
        'generated_text': generated_text,
        'prompt': prompt,
        'steps': steps,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p
    }


def plot_probability_distributions(
    step_info: Dict,
    num_top_tokens: int = 20
) -> plt.Figure:
    """
    Plot the probability distribution transformations for a single generation step.
    Shows how temperature, top-k, and top-p affect the distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f'Probability Distribution Transformations → Selected: "{step_info["token"]}"',
                 fontsize=16, fontweight='bold', y=0.99)

    # Get top tokens indices for consistent x-axis (based on original probs)
    top_indices = np.argsort(step_info['original_probs'])[-num_top_tokens:][::-1]

    # Get token labels for x-axis
    tokenizer = step_info.get('tokenizer')
    if tokenizer:
        token_labels = [
            tokenizer.decode([idx]).replace('Ġ', ' ').strip()[:15]  # Limit to 15 chars
            for idx in top_indices
        ]
    else:
        token_labels = [f"T{i}" for i in range(len(top_indices))]

    def plot_probs(ax, probs, title, color):
        top_probs = probs[top_indices]
        x_pos = np.arange(len(top_probs))

        bars = ax.bar(x_pos, top_probs, color=color, alpha=0.7, edgecolor='black')

        # Highlight selected token if in top tokens
        selected_idx = step_info['token_id']
        if selected_idx in top_indices:
            selected_pos = np.where(top_indices == selected_idx)[0][0]
            bars[selected_pos].set_color('red')
            bars[selected_pos].set_alpha(1.0)

        # Set x-axis with token labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Tokens (sorted by original probability)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, max(top_probs) * 1.1 if max(top_probs) > 0 else 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_probs)):
            if val > 0.01:  # Only show labels for bars with >1% probability
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=0)

    # Plot each stage - all showing probabilities
    plot_probs(axes[0, 0], step_info['original_probs'],
               '1. Original Probabilities (Softmax)', 'skyblue')

    plot_probs(axes[0, 1], step_info['temp_probs'],
               f'2. After Temperature ({step_info.get("temp", 1.0):.1f})', 'lightgreen')

    plot_probs(axes[1, 0], step_info['topk_probs'],
               f'3. After Top-K Filter ({step_info.get("top_k", "∞")})', 'orange')

    plot_probs(axes[1, 1], step_info['final_probs'],
               f'4. After Top-P Filter ({step_info.get("top_p", 1.0):.2f})', 'lightcoral')

    plt.tight_layout()
    return fig


def plot_sampling_comparison(results: Dict[str, Dict]) -> plt.Figure:
    """
    Compare generated text from different sampling configurations.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    configs = list(results.keys())
    y_pos = np.arange(len(configs))

    # Extract number of tokens generated for each config
    lengths = [len(res['steps']) for res in results.values()]

    bars = ax.barh(y_pos, lengths, color='steelblue', alpha=0.7, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontsize=10)
    ax.set_xlabel('Number of Tokens Generated', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Generation Lengths', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, length in zip(bars, lengths):
        ax.text(length, bar.get_y() + bar.get_height()/2,
               f' {length}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


# Main App UI
st.title("🎲 LLM Sampling Parameters Visualizer")
st.markdown("""
Explore how **Temperature**, **Top-K**, and **Top-P** sampling parameters affect text generation!

This app visualizes the probability distributions at each generation step and shows how different
sampling strategies transform these distributions.
""")

# Sidebar for configuration
# Use only the smaller TinyStories-8M model
model_name = "roneneldan/TinyStories-8M"

# Auto-load on first run
if st.session_state.extractor is None:
    with st.spinner(f"Loading {model_name}..."):
        try:
            st.session_state.extractor = AttentionExtractor(model_name=model_name)
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {str(e)}")
            st.session_state.extractor = None

# Prompt input in sidebar
st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
    **1.** Edit the **prompt** below

    **2.** Adjust **sampling parameters** (Temperature, Top-K, Top-P)

    **3.** Use the **step slider** to explore each token generation
    """)

st.sidebar.subheader("📝 Prompt")

prompt = st.sidebar.text_area(
    "Enter your prompt",
    value="There was a little dragon who",
    height=100,
    key="prompt_input",
    label_visibility="collapsed"
)

generate_button = st.sidebar.button("🚀 Generate", type="primary", use_container_width=True)

# Sampling parameters in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Sampling Parameters")

st.sidebar.markdown("**Temperature**")
temperature = st.sidebar.slider(
    "Controls randomness",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Lower = more focused/deterministic, Higher = more random/creative",
    key="temp_slider",
    label_visibility="collapsed"
)
if temperature < 0.7:
    st.sidebar.caption("🧊 Low: More deterministic")
elif temperature > 1.3:
    st.sidebar.caption("🔥 High: More random")
else:
    st.sidebar.caption("⚖️ Balanced")

st.sidebar.markdown("**Top-K**")
top_k = st.sidebar.slider(
    "Number of top tokens to consider",
    min_value=0,
    max_value=100,
    value=10,
    step=5,
    help="0 = disabled, otherwise only consider top K tokens",
    key="topk_slider",
    label_visibility="collapsed"
)
if top_k == 0:
    st.sidebar.caption("∞ All tokens")
else:
    st.sidebar.caption(f"🔢 Top {top_k}")

st.sidebar.markdown("**Top-P (Nucleus)**")
top_p = st.sidebar.slider(
    "Cumulative probability threshold",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="Only consider tokens that make up top P% of probability mass",
    key="topp_slider",
    label_visibility="collapsed"
)
if top_p < 0.5:
    st.sidebar.caption("🎯 Very focused")
elif top_p < 0.9:
    st.sidebar.caption("🎲 Moderate")
else:
    st.sidebar.caption("🌊 Wide")

# Generation settings in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Generation")

max_tokens = st.sidebar.slider(
    "Max tokens to generate",
    min_value=5,
    max_value=50,
    value=15,
    help="Number of tokens to generate",
    key="max_tokens_slider"
)

# Understanding section in sidebar
st.sidebar.markdown("---")
with st.sidebar.expander("💡 How It Works", expanded=False):
    st.markdown("""
    **The 4-Step Pipeline:**

    1. **Original Probs** - Softmax of raw logits

    2. **After Temperature** - Logits divided by temp before softmax:
       - Low (0.5): Sharper
       - High (2.0): Flatter

    3. **After Top-K** - Keep only top-K tokens, zero others

    4. **After Top-P** - Keep tokens until cumulative prob > P

    Selected token shown in **red** in plots.
    """)

# Main content tabs
tab1, tab3, tab4 = st.tabs(["📊 Visualization", "📚 Learn More", "ℹ️ Model Info"])

with tab1:
    # Auto-generate on slider changes or button click
    current_params = (prompt, temperature, top_k, top_p, max_tokens)
    slider_params = (temperature, top_k, top_p, max_tokens)

    # Initialize previous params if not exists
    if 'previous_params' not in st.session_state:
        st.session_state.previous_params = None
    if 'previous_slider_params' not in st.session_state:
        st.session_state.previous_slider_params = None
    if 'previous_prompt' not in st.session_state:
        st.session_state.previous_prompt = None

    # Check if we need to regenerate
    # Generate if: button clicked, OR sliders changed (not prompt), OR first load
    slider_changed = (st.session_state.previous_slider_params != slider_params)
    prompt_changed = (st.session_state.previous_prompt != prompt)
    first_load = 'current_result' not in st.session_state

    should_generate = (
        st.session_state.extractor is not None and
        prompt and
        (generate_button or (slider_changed and not prompt_changed) or first_load)
    )

    if should_generate:
        with st.spinner("Generating and analyzing..."):
            try:
                result = generate_with_sampling_visualization(
                    extractor=st.session_state.extractor,
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    num_tokens_to_visualize=20
                )
                st.session_state.current_result = result
                st.session_state.previous_params = current_params
                st.session_state.previous_slider_params = slider_params
                st.session_state.previous_prompt = prompt

            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display visualizations
    if 'current_result' in st.session_state and st.session_state.current_result:
        result = st.session_state.current_result

        # Select which step to visualize
        step_idx = st.slider(
            f"Select generation step (1-{len(result['steps'])})",
            min_value=0,
            max_value=len(result['steps']) - 1,
            value=0,
            help="Choose which token generation step to analyze",
            key="step_slider"
        )

        step_info = result['steps'][step_idx]
        step_info['temp'] = temperature
        step_info['top_k'] = top_k if top_k > 0 else "∞"
        step_info['top_p'] = top_p

        # Create two columns: text on left, plots on right
        col_text, col_plot = st.columns([1, 2])

        with col_text:
            st.markdown("### Generated Text")

            # Build the generated text with the selected token highlighted
            # Get all generated tokens
            all_tokens = [step['token'] for step in result['steps']]

            # Create HTML with selected token in red
            html_text = "<p style='font-size: 20px; line-height: 1.8;'>"

            # Add prompt first (not highlighted)
            html_text += f"{result['prompt']} "

            # Add generated tokens, highlighting the selected one
            for i, token in enumerate(all_tokens):
                # Clean up token display (remove leading space marker)
                display_token = token.replace('Ġ', ' ')

                if i == step_idx:
                    html_text += f"<span style='color: red; font-weight: bold; background-color: #ffe6e6; padding: 2px 4px; border-radius: 3px;'>{display_token}</span>"
                else:
                    html_text += display_token

            html_text += "</p>"
            st.markdown(html_text, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"**Step {step_idx + 1}/{len(result['steps'])}**")
            st.markdown(f"Selected token: <span style='color: red; font-weight: bold; font-size: 18px;'>\"{step_info['token']}\"</span>", unsafe_allow_html=True)
            st.markdown(f"Probability: **{step_info['selected_token_prob']:.4f}**")

            st.markdown("---")
            st.markdown("**Top alternatives:**")
            alternatives = [(token, prob) for token, prob in step_info['top_tokens'] if token != step_info['token']]
            for token, prob in alternatives[:5]:
                st.markdown(f"• \"{token}\" - {prob:.4f}")

        with col_plot:
            # Plot distributions
            fig = plot_probability_distributions(step_info, num_top_tokens=20)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close()

    elif st.session_state.extractor is None:
        st.info("⏳ Loading model, please wait...")
    elif not prompt:
        st.info("👆 Enter a prompt in the sidebar to begin")

# with tab2:
#     st.header("Compare Different Sampling Configurations")
#
#     st.markdown("""
#     Generate text with multiple sampling configurations and compare the results side-by-side.
#     """)
#
#     # Comparison prompt
#     comparison_prompt = st.text_input(
#         "Prompt for comparison",
#         value="Once upon a time",
#         key="comparison_prompt"
#     )
#
#     # Predefined configurations
#     st.subheader("Select Configurations to Compare")
#
#     configs = {
#         "Greedy (temp=0.1)": {"temperature": 0.1, "top_k": 0, "top_p": 1.0},
#         "Balanced (temp=1.0)": {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
#         "Creative (temp=1.5)": {"temperature": 1.5, "top_k": 0, "top_p": 1.0},
#         "Very Creative (temp=2.0, top-k=100)": {"temperature": 2.0, "top_k": 100, "top_p": 1.0},
#         "Top-K 50": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
#         "Top-P 0.9": {"temperature": 1.0, "top_k": 0, "top_p": 0.9},
#         "Top-K 40 + Top-P 0.95": {"temperature": 1.0, "top_k": 40, "top_p": 0.95},
#     }
#
#     selected_configs = st.multiselect(
#         "Choose configurations",
#         options=list(configs.keys()),
#         default=list(configs.keys())[:3]
#     )
#
#     comparison_max_tokens = st.slider(
#         "Max tokens for comparison",
#         min_value=10,
#         max_value=50,
#         value=20,
#         key="comparison_max_tokens"
#     )
#
#     if st.button("Generate Comparisons", type="primary", key="compare_button"):
#         if not comparison_prompt:
#             st.error("Please enter a prompt")
#         elif st.session_state.extractor is None:
#             st.error("Please load a model first")
#         elif not selected_configs:
#             st.error("Please select at least one configuration")
#         else:
#             with st.spinner("Generating comparisons..."):
#                 try:
#                     comparison_results = {}
#
#                     for config_name in selected_configs:
#                         config = configs[config_name]
#                         result = generate_with_sampling_visualization(
#                             extractor=st.session_state.extractor,
#                             prompt=comparison_prompt,
#                             temperature=config['temperature'],
#                             top_k=config['top_k'],
#                             top_p=config['top_p'],
#                             max_new_tokens=comparison_max_tokens,
#                             num_tokens_to_visualize=20
#                         )
#                         comparison_results[config_name] = result
#
#                     st.session_state.comparison_results = comparison_results
#
#                     # Display results
#                     st.header("Comparison Results")
#
#                     for config_name, result in comparison_results.items():
#                         st.subheader(config_name)
#                         st.info(f"**Generated:** {result['generated_text']}")
#                         st.caption(f"Tokens: {len(result['steps'])} | Temp: {result['temperature']} | Top-K: {result['top_k']} | Top-P: {result['top_p']}")
#                         st.markdown("---")
#
#                 except Exception as e:
#                     st.error(f"Error during comparison: {str(e)}")
#                     import traceback
#                     st.code(traceback.format_exc())

with tab4:
    st.header("Model Information")

    if st.session_state.extractor is not None:
        config = st.session_state.extractor.model.config

        st.markdown(f"""
        ### TinyStories-8M

        A small language model trained on simple children's stories.

        #### Architecture Specifications

        - **Total Parameters:** ~8,116,224 (≈8M)
        - **Model Type:** Causal (Autoregressive) Transformer
        - **Architecture Style:** GPT-2 based

        #### Transformer Configuration

        - **Number of Layers:** {config.num_hidden_layers} transformer blocks
        - **Attention Heads per Layer:** {config.num_attention_heads} heads
        - **Hidden Size:** {config.hidden_size} dimensions
        - **Head Dimension:** {config.hidden_size // config.num_attention_heads} (hidden_size / num_heads)
        - **Vocabulary Size:** {config.vocab_size:,} tokens
        - **Maximum Sequence Length:** 2,048 tokens

        #### Memory & Performance

        - **Model weights:** ~32 MB (8M params × 4 bytes/float32)
        - **Single token generation:** 10-30ms (CPU)
        - **Suitable for:** Real-time interactive applications

        #### Training Details

        The TinyStories models were trained on a synthetic dataset of simple children's stories
        generated to contain only words that a typical 3-4 year old would understand.

        #### References

        **TinyStories Paper:**
        - Eldan, R., & Li, Y. (2023). "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
        - arXiv:2305.07759
        - [https://arxiv.org/abs/2305.07759](https://arxiv.org/abs/2305.07759)
        """)
    else:
        st.info("Model not loaded yet. Please wait...")

with tab3:
    st.header("Understanding Sampling Parameters")

    st.markdown("""
    ## What are Sampling Parameters?

    When a language model generates text, it doesn't simply pick the most likely next word.
    Instead, it uses **sampling strategies** to introduce controlled randomness, making the output
    more diverse and interesting.

    ### 🌡️ Temperature

    **What it does:** Controls the randomness of predictions by scaling the logits before applying softmax.

    **Mathematical formulation:**

    $$P(x_i) = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}$$

    where $z_i$ are the logits and $T$ is the temperature.

    **Effects:**
    - **Low temperature (< 1.0)**: Makes the model more confident and deterministic
      - Probability distribution becomes sharper (peaked)
      - Model picks high-probability tokens more often
      - Good for factual, focused responses

    - **High temperature (> 1.0)**: Makes the model more random and creative
      - Probability distribution becomes flatter
      - Model explores lower-probability tokens
      - Good for creative writing, brainstorming

    **Example:**
    - T=0.1: "The cat is sleeping" (predictable)
    - T=1.5: "The cat is dreaming about quantum physics" (creative)

    ---

    ### 🔢 Top-K Sampling

    **What it does:** Restricts sampling to the K most likely tokens at each step.

    **Mathematical formulation:**

    Let V_K be the set of top-K tokens. The filtered probability is:

    - P'(x_i) = P(x_i) if x_i ∈ V_K
    - P'(x_i) = 0 otherwise

    Then renormalize by dividing by the sum over V_K tokens.

    **Algorithm:**
    1. Sort all tokens by probability
    2. Keep only the top K tokens
    3. Set probability of all other tokens to 0
    4. Renormalize and sample from this restricted set

    **Effects:**
    - **K=1**: Greedy decoding (always pick most likely)
    - **K=50**: Moderate diversity (common setting)
    - **K=∞ (or 0)**: No restriction (use all tokens)

    **Advantages:**
    - Simple and effective
    - Prevents sampling very unlikely tokens
    - Fixed computational cost

    **Disadvantages:**
    - Fixed K doesn't adapt to distribution shape
    - May be too restrictive when distribution is flat
    - May be too permissive when distribution is peaked

    ---

    ### 🎯 Top-P (Nucleus) Sampling

    **What it does:** Dynamically selects the smallest set of tokens whose cumulative probability exceeds P.

    **Mathematical formulation:**

    Define the nucleus V_P as the smallest set where the cumulative probability ≥ p (tokens sorted by descending probability).

    The final probability becomes:
    - P_top-p(x_i) = P(x_i) / Σ(P(x_j) for j ∈ V_P) if x_i ∈ V_P
    - P_top-p(x_i) = 0 otherwise

    **Algorithm:**
    1. Sort tokens by probability (descending)
    2. Compute cumulative probability
    3. Keep tokens until cumulative probability exceeds P
    4. Renormalize and sample from this set

    **Effects:**
    - **P=0.5**: Very focused (only highly probable tokens)
    - **P=0.9**: Balanced (common setting)
    - **P=1.0**: No restriction (all tokens)

    **Advantages:**
    - Adapts to distribution shape
    - More permissive when distribution is flat
    - More restrictive when distribution is peaked
    - Generally produces higher quality text than top-k

    **Disadvantages:**
    - Variable computational cost
    - Can still sample very unlikely tokens in flat distributions

    ---

    ### 🎨 Combining Parameters

    You can combine temperature, top-k, and top-p for fine-grained control:

    1. **Apply temperature** to logits → adjust distribution shape
    2. **Apply top-k** filtering → restrict to K tokens
    3. **Apply top-p** filtering → further restrict by cumulative probability
    4. **Sample** from final distribution

    **Common combinations:**
    - **Factual/Precise**: temp=0.7, top-k=40, top-p=0.9
    - **Balanced**: temp=1.0, top-k=50, top-p=0.95
    - **Creative**: temp=1.3, top-k=0, top-p=0.95
    - **Very Creative**: temp=1.5, top-k=100, top-p=1.0


    """)