"""
Streamlit app for analyzing attention patterns during text generation.
Shows how each generated token attends to ALL tokens in the prompt.
"""

import streamlit as st
import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_loader import AttentionExtractor
from visualizer import AttentionVisualizer
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="Show Me Your Attention",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'extractor' not in st.session_state:
    st.session_state.extractor = None

if 'viz' not in st.session_state:
    st.session_state.viz = AttentionVisualizer()

if 'generation_result' not in st.session_state:
    st.session_state.generation_result = None

if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = None

if 'qkv_data' not in st.session_state:
    st.session_state.qkv_data = None

# Title and description
st.title("Show Me Your Attention")
st.header("(...at least once upon a time.)")
st.markdown("""
Welcome to a place where you can generate fairytales and observe attention mechanism at the same time. Fancy matrices included!
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Model selection - TinyStories 8M only
model_name = "roneneldan/TinyStories-8M"
st.sidebar.info(f"**Model:** {model_name}")

# Load model button
if st.sidebar.button("🚀 Load Model") or st.session_state.extractor is None:
    if model_name:
        with st.spinner(f"Loading {model_name}..."):
            try:
                st.session_state.extractor = AttentionExtractor(model_name=model_name)
                st.sidebar.success(f"✅ Loaded {model_name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load model: {str(e)}")
                st.session_state.extractor = None
    else:
        st.sidebar.error("Please select a model")

# Generation parameters
st.sidebar.header("🎛️ Generation Settings")
max_tokens = st.sidebar.slider("Max tokens to generate", 5, 60, 30, help="Number of tokens to generate (fewer = cleaner visualization)")
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

# Fixed figure sizes and settings
fig_width = 18
fig_height = 12
show_probabilities = True  # Always show token probabilities

# Main content
tab1, tab2 = st.tabs(["📝 Analysis", "🤖 Model Info"])

with tab1:

    # What this app does and how it works section
    with st.expander("📚 About & How to Use", expanded=False):
        st.markdown("""
        ### What This App Does

        This tool analyzes **attention patterns during text generation**. It provides three complementary views:

        1. **Generation Attention**: How each newly generated token attends back to the original prompt
        2. **Self-Attention (Q, K, V)**: How prompt tokens attend to each other, revealing the internal attention mechanism with detailed mathematical breakdown
        3. **Complete Attention Matrix**: Full self-attention matrix showing how all tokens (prompt + generated) attend to each other in one comprehensive heatmap

        ---

        ### Quick Start Guide

        1. **Enter a prompt**: Type any text (e.g., "The cat is") or select an example
        2. **Generate text**: Click "Generate & Visualize" - the model continues your prompt
        3. **Explore visualizations**:
           - Generation attention bars (how generated words look back at prompt)
           - Q, K, V matrices (internal attention mechanics)
           - Complete attention matrix (the big picture!)

        """)

    st.subheader("Enter Your Prompt")

    # Example prompts
    example = st.selectbox(
        "Load Example",
        [
            "Custom",
            "The cat is",
            "Once upon a time",
            "A little girl named",
            "The dragon was very",
            "In the forest there"
        ]
    )

    if example != "Custom":
        default_prompt = example
    else:
        default_prompt = ""

    prompt = st.text_input(
        "Write start of your fairytale (e.g. 'once upon a time')",
        value=default_prompt,
        help="Enter a prompt to generate from"
    )

    # Use last layer by default for generation
    layer_mode = "last"

    if st.button("🚀 Generate & Visualize", type="primary"):
        if not prompt:
            st.error("Please enter a prompt")
        elif st.session_state.extractor is None:
            st.error("Please load a model first (use sidebar)")
        else:
            with st.spinner("Generating text and analyzing attention..."):
                try:
                    # Generate text with attention tracking to all prompt tokens
                    result = st.session_state.extractor.generate_with_attention_to_all_prompt_tokens(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        layer_mode=layer_mode
                    )

                    # Store in session state
                    st.session_state.generation_result = result
                    st.session_state.current_prompt = prompt

                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display results if they exist in session state
    if st.session_state.generation_result is not None:
        result = st.session_state.generation_result
        prompt = st.session_state.current_prompt

        # Display results
        st.header("📊 Results")

        # Show prompt
        st.markdown("**Original Prompt:**")
        st.info(prompt)

        # Show generated text
        st.markdown("**Generated Text:**")
        st.success(result['generated_text'])

        # Show main visualization
        st.header("🎯 Generation Attention Visualization")

        with st.expander("📖 What does this show?", expanded=False):
            st.markdown("""
            ### Generated Tokens → Prompt Tokens Attention

            **What it shows:** How each newly generated word attends back to the original prompt during text generation.

            **Dimensions:** `[num_generated_tokens × num_prompt_tokens]`

            **Purpose:** Understanding which prompt words influence each generated word as the model creates new text.

            **Example:** If your prompt is "The cat is" and the model generates "sleeping on the":
            - See how "sleeping" attends to ["The", "cat", "is"]
            - See how "on" attends to ["The", "cat", "is"]
            - See how "the" attends to ["The", "cat", "is"]

            **Reading the visualization:**
            - Each subplot shows one prompt word
            - X-axis: Generated tokens (in order)
            - Bar height = attention strength
            - Color = relative attention (darker = stronger within that subplot)
            """)

        # Layer selection for generation attention
        layer_options = {
            "Last layer only (Layer 7)": "last",
            "Average across all layers": "average",
            "Layer 0 (Earliest)": "layer_0",
            "Layer 1": "layer_1",
            "Layer 2": "layer_2",
            "Layer 3": "layer_3",
            "Layer 4": "layer_4",
            "Layer 5": "layer_5",
            "Layer 6": "layer_6",
            "Layer 7 (Last)": "layer_7"
        }
        layer_selection_display = st.selectbox(
            "Which layer(s) to analyze",
            options=list(layer_options.keys()),
            index=0,  # Default to "Last layer only"
            help="Choose which attention layer(s) to visualize. Note: This was used during generation.",
            key="gen_layer_display"
        )

        st.markdown("""
        **Attention Calculation:**

        For each generated token $g_i$ attending to prompt token $p_j$:

        $$\\text{{Attention}}(g_i, p_j) = \\frac{{1}}{{L \\times H}} \\sum_{{\\ell=1}}^{{L}} \\sum_{{h=1}}^{{H}} \\text{{softmax}}\\left(\\frac{{Q_{i}^{{(\\ell,h)}} \\cdot K_{j}^{{(\\ell,h)}}}}{{\\sqrt{{d_k}}}}\\right)$$

        where $L$ = layers, $H$ = heads per layer, $d_k$ = head dimension
        """)

        st.markdown("(**Token probabilities** shown on x-axis indicate how confident the model was when generating each token (higher = more certain).)")

        # Create and display visualization
        fig = plt.figure(figsize=(fig_width, fig_height))
        st.session_state.viz.plot_generated_attention_to_all_prompt_tokens(
            generation_result=result,
            save_path=None,
            figsize=(fig_width, fig_height),
            show_probabilities=show_probabilities
        )

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()

        # Show attention matrix
        with st.expander("📊 Attention Matrix (Raw Values)"):
            st.markdown("**Rows:** Generated tokens | **Columns:** Prompt tokens")

            # Create attention matrix
            attention_matrix = np.array(result['attention_to_prompt'])

            # Display as dataframe
            import pandas as pd
            df = pd.DataFrame(
                attention_matrix,
                index=[f"Gen[{i}]: {tok}" for i, tok in enumerate(result['generated_tokens'])],
                columns=result['prompt_tokens']
            )
            st.dataframe(df.style.background_gradient(cmap='viridis', axis=1), use_container_width=True)

        # Q, K, V Visualization Section
        st.markdown("---")
        st.header("🔬 Prompt Self-Attention Mechanism")

        with st.expander("📖 What does this show?", expanded=False):
            st.markdown("""
            ### Prompt Tokens → Prompt Tokens (Self-Attention)

            **What it shows:** Self-attention within the prompt itself - how prompt tokens attend to each other BEFORE generation starts.

            **Dimensions:** `[num_prompt_tokens × num_prompt_tokens]`

            **Purpose:** Understanding the internal attention mechanism and demonstrating the mathematical mechanics of how attention works at a specific layer and head.

            **Example:** If your prompt is "The cat is":
            - See how "The" attends to ["The", "cat", "is"]
            - See how "cat" attends to ["The", "cat", "is"]
            - See how "is" attends to ["The", "cat", "is"]

            **Key Difference from Generation Attention:**
            - **Generation Attention** (above): Shows how NEW generated words look back at the prompt
            - **Self-Attention** (this section): Shows how the prompt understands itself internally

            **What you'll see:**
            - Complete attention pipeline: Q × K^T → Scores → Softmax → Attention Weights → Attention × V → Output
            - Raw Q, K, V matrices that form the foundation of attention
            - Step-by-step mathematical transformations with visual separators
            """)

        with st.expander("🔬 Explore Q, K, V Matrices", expanded=True):
            st.markdown("""
            **The three fundamental matrices:**
            - **Query (Q)**: What each token is "looking for"
            - **Key (K)**: What each token "contains"
            - **Value (V)**: What information each token "carries"
            """)

            st.markdown("""
            **Q, K, V Computation:**

            For each token's hidden state $h_i$ at layer $\\ell$, head $h$:

            $$Q_i^{(\\ell,h)} = W_Q^{(\\ell,h)} \\cdot h_i$$

            $$K_i^{(\\ell,h)} = W_K^{(\\ell,h)} \\cdot h_i$$

            $$V_i^{(\\ell,h)} = W_V^{(\\ell,h)} \\cdot h_i$$

            where $W_Q, W_K, W_V$ are learned projection matrices, and $h_i \\in \\mathbb{R}^{d_{\\text{model}}}$
            """)

            st.markdown("""
            **Attention Calculation:**

            $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q \\cdot K^T}{\\sqrt{d_k}}\\right) \\cdot V$$

            The attention output is then: $\\text{Output}_i = \\sum_j \\text{Attention}(i,j) \\cdot V_j$

            where $d_k$ is the head dimension (key dimension).
            """)

            # Controls for Q, K, V selection
            col1, col2 = st.columns(2)
            with col1:
                qkv_layer = st.selectbox(
                    "Select layer for Q/K/V",
                    options=list(range(8)),
                    index=7,
                    help="Which layer to extract Q, K, V from"
                )
            with col2:
                qkv_head = st.slider(
                    "Select attention head",
                    min_value=0,
                    max_value=15,
                    value=0,
                    help="Which attention head to visualize (0-15)"
                )

            # Auto-extract Q, K, V on first load or when layer changes
            if st.session_state.qkv_data is None or st.session_state.qkv_data.get('layer_idx') != qkv_layer:
                with st.spinner(f"Extracting Q, K, V matrices from Layer {qkv_layer}..."):
                    try:
                        st.session_state.qkv_data = st.session_state.extractor.extract_qkv_matrices(
                            prompt=prompt,
                            layer_idx=qkv_layer
                        )
                    except Exception as e:
                        st.error(f"Error extracting Q, K, V: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            # Display Q, K, V results if they exist
            if st.session_state.qkv_data is not None:
                qkv_data = st.session_state.qkv_data

                # Show info
                st.markdown(f"""
                **Matrix Dimensions:**
                - Number of tokens: {len(qkv_data['tokens'])}
                - Number of heads: {qkv_data['num_heads']}
                - Head dimension: {qkv_data['head_dim']}
                - Each matrix shape: [{qkv_data['num_heads']}, {len(qkv_data['tokens'])}, {qkv_data['head_dim']}]
                """)

                # Visualize Q, K, V for selected head
                st.subheader(f"Complete Attention Calculation for Head {qkv_head}")
                st.markdown("""
                This visualization shows the complete attention mechanism calculation:

                **Row 1 - Attention Weights Calculation:**
                1. **Q × Kᵀ**: Compute similarity scores between queries and keys
                2. **Scores**: Raw attention scores (scaled by √d)
                3. **Softmax**: Normalize to create attention weights (sum to 1)

                **Row 2 - Final Output:**
                4. **Attention × V**: Use weights to combine value vectors → **Output**

                Each multiplication (×) is clearly shown with the result matrix.
                """)
                fig_qkv = plt.figure(figsize=(24, 12))
                st.session_state.viz.plot_qkv_matrices(
                    qkv_data=qkv_data,
                    head_idx=qkv_head
                )
                buf_qkv = io.BytesIO()
                plt.savefig(buf_qkv, format='png', dpi=150, bbox_inches='tight')
                buf_qkv.seek(0)
                st.image(buf_qkv, use_container_width=True)
                plt.close()

        # Full Attention Matrix Section
        st.markdown("---")
        st.header("🌐 Complete Attention Matrix")

        with st.expander("📖 What does this show?", expanded=False):
            st.markdown("""
            ### Full Self-Attention: All Tokens → All Tokens

            **What it shows:** Complete attention matrix showing how every token (both prompt and generated) attends to every other token.

            **Dimensions:** `[total_tokens × total_tokens]` where total_tokens = prompt_tokens + generated_tokens

            **Purpose:** See the complete picture of how the entire sequence attends to itself during generation.

            **Reading the matrix:**
            - **Rows**: Query tokens (what's attending)
            - **Columns**: Key tokens (what's being attended to)
            - **Upper-left quadrant**: Prompt attending to prompt (self-attention)
            - **Upper-right quadrant**: Prompt attending to generated tokens (impossible due to causal masking - will be zero)
            - **Lower-left quadrant**: Generated tokens attending to prompt
            - **Lower-right quadrant**: Generated tokens attending to each other
            - **Diagonal line**: Each token attending to itself

            **Note:** Due to causal (autoregressive) masking, tokens can only attend to previous tokens, creating a triangular pattern.
            """)

        with st.expander("🔬 View Full Attention Matrix", expanded=True):
            # Get full sequence tokens
            all_tokens = result['prompt_tokens'] + result['generated_tokens']
            n_prompt = len(result['prompt_tokens'])
            n_generated = len(result['generated_tokens'])
            n_total = n_prompt + n_generated

            st.markdown(f"""
            **Matrix Information:**
            - Total tokens: {n_total} ({n_prompt} prompt + {n_generated} generated)
            - Matrix size: {n_total} × {n_total}
            - Prompt tokens: positions 0-{n_prompt-1}
            - Generated tokens: positions {n_prompt}-{n_total-1}
            """)

            # Layer selection for full matrix
            col1, col2 = st.columns([3, 1])
            with col1:
                full_matrix_layer = st.selectbox(
                    "Select layer for full attention matrix",
                    options=list(range(8)),
                    index=7,
                    help="Which layer to visualize the complete attention matrix",
                    key="full_matrix_layer"
                )

            with st.spinner("Computing full attention matrix..."):
                try:
                    # Re-run the full sequence through the model to get complete attention
                    full_text = prompt + result['generated_text'][len(prompt):]
                    inputs = st.session_state.extractor.tokenizer(full_text, return_tensors="pt")
                    inputs = {k: v.to(st.session_state.extractor.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = st.session_state.extractor.model(**inputs)

                    # Get attention from specified layer
                    # Shape: [batch, num_heads, seq_len, seq_len]
                    layer_attention = outputs.attentions[full_matrix_layer][0]  # Remove batch dim

                    # Average across all heads
                    full_attention_matrix = layer_attention.mean(dim=0).cpu().numpy()

                    # Create visualization
                    fig, ax = plt.subplots(figsize=(16, 14))

                    # Plot heatmap
                    im = ax.imshow(full_attention_matrix, cmap='viridis', aspect='auto', interpolation='nearest')

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Attention Weight', fontsize=12, weight='bold')

                    # Set ticks and labels
                    ax.set_xticks(range(n_total))
                    ax.set_yticks(range(n_total))
                    ax.set_xticklabels(all_tokens, rotation=90, ha='right', fontsize=8)
                    ax.set_yticklabels(all_tokens, fontsize=8)

                    # Remove grid lines
                    ax.grid(False)

                    # Add dividing lines to separate prompt from generated (not crossing lines)
                    ax.axhline(y=n_prompt-0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
                    ax.axvline(x=n_prompt-0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)

                    # Add labels for quadrants
                    ax.text(n_prompt/2, -1.5, 'PROMPT', ha='center', va='bottom',
                           fontsize=11, weight='bold', color='darkblue')
                    ax.text(n_prompt + n_generated/2, -1.5, 'GENERATED', ha='center', va='bottom',
                           fontsize=11, weight='bold', color='darkgreen')
                    ax.text(-1.5, n_prompt/2, 'PROMPT', ha='right', va='center', rotation=90,
                           fontsize=11, weight='bold', color='darkblue')
                    ax.text(-1.5, n_prompt + n_generated/2, 'GENERATED', ha='right', va='center', rotation=90,
                           fontsize=11, weight='bold', color='darkgreen')

                    ax.set_title(f'Complete Attention Matrix - Layer {full_matrix_layer}\n(Averaged across all {layer_attention.shape[0]} heads)',
                               fontsize=14, weight='bold', pad=20)
                    ax.set_xlabel('Key Tokens (attending TO)', fontsize=12, weight='bold')
                    ax.set_ylabel('Query Tokens (attending FROM)', fontsize=12, weight='bold')

                    plt.tight_layout()

                    # Display
                    buf_full = io.BytesIO()
                    plt.savefig(buf_full, format='png', dpi=150, bbox_inches='tight')
                    buf_full.seek(0)
                    st.image(buf_full, use_container_width=True)
                    plt.close()

                except Exception as e:
                    st.error(f"Error generating full attention matrix: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

with tab2:

    st.markdown("""

    **TinyStories-8M** is a small language model trained on simple children's stories.

    #### Architecture Specifications

    - **Total Parameters:** 8,116,224 (≈8M)
    - **Trainable Parameters:** 8,116,224 (100% trainable)
    - **Model Type:** Causal (Autoregressive) Transformer
    - **Architecture Style:** GPT-2 based

    #### Transformer Configuration

    - **Number of Layers:** 8 transformer blocks
    - **Attention Heads per Layer:** 16 heads
    - **Hidden Size:** 256 dimensions
    - **Head Dimension:** 16 (hidden_size / num_heads = 256 / 16)
    - **Vocabulary Size:** 50,257 tokens
    - **Maximum Sequence Length:** 2,048 tokens
    - **Context Window:** 2,048 tokens

    #### Parameter Breakdown

    Each transformer layer contains:
    - **Self-Attention Module:**
      - Query (Q), Key (K), Value (V) projections: 3 × (256 × 256) = 196,608 params
      - Output projection: 256 × 256 = 65,536 params
      - Total per layer: ~262K params

    - **Feed-Forward Network:**
      - First linear layer: 256 × 1,024 = 262,144 params (4× expansion)
      - Second linear layer: 1,024 × 256 = 262,144 params
      - Total per layer: ~524K params

    - **Layer Normalization:** 2 × 256 = 512 params per layer

    **Total per transformer block:** ~787K parameters × 8 layers = ~6.3M

    **Embedding layers:**
    - Token embeddings: 50,257 × 256 = ~12.9M params
    - Position embeddings: 2,048 × 256 = ~0.5M params

    #### Training Details

    The TinyStories models were trained on a synthetic dataset of simple children's stories
    generated to contain only words that a typical 3-4 year old would understand. 

    #### Attention Mechanism Details

    **Multi-Head Self-Attention:**
    - Each layer has 16 attention heads operating in parallel
    - Each head learns different attention patterns (e.g., syntax, semantics, positional)
    - Attention scores are computed as: `softmax(Q @ K^T / sqrt(16))`
    - Output is weighted sum of Values: `Attention @ V`

    **Causal Masking:**
    - Attention is masked to prevent looking at future tokens
    - This ensures autoregressive generation (left-to-right)
    - Each position can only attend to itself and previous positions

    #### Performance Characteristics

    **Memory Usage:**
    - Model weights: ~32 MB (8M params × 4 bytes/float32)
    - Activations (50 tokens): ~10 MB
    - Total GPU/CPU memory: ~50-100 MB

    **Inference Speed (CPU):**
    - Single token generation: 10-30ms
    - Attention extraction overhead: <5ms
    - Suitable for real-time interactive applications

    #### Comparison to Other Models

    | Model | Parameters | Layers | Heads | Hidden Size |
    |-------|-----------|--------|-------|-------------|
    | **TinyStories-8M** | 8M | 8 | 16 | 256 |
    | TinyStories-33M | 33M | 8 | 12 | 768 |
    | GPT-2 Small | 124M | 12 | 12 | 768 |
    | GPT-2 Medium | 355M | 24 | 16 | 1024 |

    ---

    ### References

    **TinyStories Paper:**
    - Eldan, R., & Li, Y. (2023). "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
    - arXiv:2305.07759
    - [https://arxiv.org/abs/2305.07759](https://arxiv.org/abs/2305.07759)

    """)
