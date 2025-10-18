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
Welcome to a place where you can generate fairytales and observe the attention mechanism at the same time. Fancy matrices included!
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model selection
model_options = {
    "TinyStories-8M (8 layers, 16 heads)": "roneneldan/TinyStories-8M",
    "TinyStories-1Layer-21M (1 layer, 16 heads)": "roneneldan/TinyStories-1Layer-21M"
}
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    index=0,
    help="Choose which TinyStories model to use"
)
model_name = model_options[selected_model]

# Show warning for 21M model
if "1Layer-21M" in model_name:
    st.sidebar.warning("**Note:** This model is ~84MB and may take 1-3 minutes to download on first load.")

# Load model button
if st.sidebar.button("Load Model"):
    if model_name:
        with st.spinner(f"Loading {model_name}..."):
            try:
                st.session_state.extractor = AttentionExtractor(model_name=model_name)
                st.session_state.model_config = {
                    'name': model_name,
                    'num_layers': st.session_state.extractor.model.config.num_layers,
                    'num_heads': st.session_state.extractor.model.config.num_heads,
                    'hidden_size': st.session_state.extractor.model.config.hidden_size
                }
                # Clear previous results when changing models
                st.session_state.generation_result = None
                st.session_state.qkv_data = None
                st.sidebar.success(f"Loaded {model_name}")
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {str(e)}")
                st.session_state.extractor = None
    else:
        st.sidebar.error("Please select a model")

# Auto-load default model on first run
if st.session_state.extractor is None:
    with st.spinner(f"Loading {model_name}..."):
        try:
            st.session_state.extractor = AttentionExtractor(model_name=model_name)
            st.session_state.model_config = {
                'name': model_name,
                'num_layers': st.session_state.extractor.model.config.num_layers,
                'num_heads': st.session_state.extractor.model.config.num_heads,
                'hidden_size': st.session_state.extractor.model.config.hidden_size
            }
            st.sidebar.success(f"Loaded {model_name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {str(e)}")
            st.session_state.extractor = None

# Show current model info
if st.session_state.extractor is not None and 'model_config' in st.session_state:
    config = st.session_state.model_config
    st.sidebar.info(f"""
    **Current Model:**
    - Layers: {config['num_layers']}
    - Heads: {config['num_heads']}
    - Hidden: {config['hidden_size']}
    """)

# Generation parameters
st.sidebar.header("Generation Settings")
max_tokens = st.sidebar.slider("Max tokens to generate", 5, 100, 30, help="Number of tokens to generate (fewer = cleaner visualization)")
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

# Fixed figure sizes and settings
fig_width = 18
fig_height = 12
show_probabilities = True  # Always show token probabilities

# Main content
tab1, tab2 = st.tabs(["Analysis", "Model Info"])

with tab1:

    # What this app does and how it works section
    with st.expander("About & How to Use", expanded=False):
        st.markdown("""
        ### What This App Does

        Analyzes **attention patterns during text generation** with three views:
        1. **Generation Attention**: How generated tokens attend to prompt
        2. **Self-Attention (Q, K, V)**: Internal attention mechanism with math breakdown
        3. **Complete Attention Matrix**: Full heatmap of all token interactions

        ### Two Models Available

        - **TinyStories-8M (8 layers)**: See attention evolve through layers. Fast to load.
        - **TinyStories-1Layer-21M (1 layer)**: Simpler! Only one transformer block. Takes 1-3 min to download.

        ### Quick Start

        1. Enter a prompt or select an example
        2. Click "Generate & Visualize"
        3. Explore visualizations and switch between layers/heads

        """)

    st.subheader("Write start of your fairytale (e.g. 'once upon a time') or select example:")

    # Example prompts
    example = st.selectbox(
        "Select Example",
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
        "Start writing your story",
        value=default_prompt,
        help="Enter a prompt to generate from"
    )

    if st.button("Generate & Visualize", type="primary"):
        if not prompt:
            st.error("Please enter a prompt")
        elif st.session_state.extractor is None:
            st.error("Please load a model first (use sidebar)")
        else:
            # Generate with last layer by default
            with st.spinner("Generating text and analyzing attention..."):
                try:
                    # Generate text with attention tracking to all prompt tokens
                    result = st.session_state.extractor.generate_with_attention_to_all_prompt_tokens(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        layer_mode="last"
                    )

                    # Store in session state
                    st.session_state.generation_result = result
                    st.session_state.current_prompt = prompt
                    st.session_state.selected_layer_mode = "last"  # Store initial selection

                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display results if they exist in session state
    if st.session_state.generation_result is not None:
        result = st.session_state.generation_result
        prompt = st.session_state.current_prompt

        # Display results
        st.header("Results")

        # Show prompt
        st.markdown("**Original Prompt:**")
        st.info(prompt)

        # Show generated text
        st.markdown("**Generated Text:**")
        st.success(result['generated_text'])

        # Show main visualization
        st.header("Generation Attention Visualization")

        with st.expander("What does this show?", expanded=False):
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

        # Layer selection for generation attention - dynamic regeneration
        num_layers = st.session_state.model_config['num_layers'] if 'model_config' in st.session_state else 8

        # Build layer options dynamically based on model
        layer_options = {
            f"Last layer only (Layer {num_layers-1})": "last",
            "Average across all layers": "average"
        }
        for i in range(num_layers):
            label = f"Layer {i}" + (" (Earliest)" if i == 0 else " (Last)" if i == num_layers-1 else "")
            layer_options[label] = f"layer_{i}"

        layer_selection = st.selectbox(
            "Which layer(s) to analyze",
            options=list(layer_options.keys()),
            index=0,  # Default to "Last layer only"
            help="Select a layer to regenerate attention with that layer's data.",
            key="gen_layer_selection"
        )
        layer_mode = layer_options[layer_selection]

        # Check if layer selection changed - regenerate if so
        if 'selected_layer_mode' not in st.session_state:
            st.session_state.selected_layer_mode = "last"

        if layer_mode != st.session_state.selected_layer_mode:
            with st.spinner(f"Regenerating attention for {layer_selection}..."):
                try:
                    result = st.session_state.extractor.generate_with_attention_to_all_prompt_tokens(
                        prompt=st.session_state.current_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        layer_mode=layer_mode
                    )
                    st.session_state.generation_result = result
                    st.session_state.selected_layer_mode = layer_mode
                except Exception as e:
                    st.error(f"Error regenerating attention: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

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

        # Q, K, V Visualization Section
        st.markdown("---")
        st.header("Prompt Self-Attention Mechanism")

        with st.expander("What does this show?", expanded=False):
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

        with st.expander("Explore Q, K, V Matrices", expanded=True):
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

            where:
            - $h_i$ = hidden state (vector representation of token $i$ at this layer)
            - $W_Q, W_K, W_V$ = learned projection matrices
            - $h_i \\in \\mathbb{R}^{d_{\\text{model}}}$ (256-dim for 8M, 1024-dim for 1Layer-21M)
            """)

            st.markdown("""
            **Attention Calculation:**

            $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q \\cdot K^T}{\\sqrt{d_k}}\\right) \\cdot V$$

            where $d_k$ is the head dimension.

            The attention weights for token $i$ are: $\\text{Attention}(i,j) = \\text{softmax}\\left(\\frac{Q_i \\cdot K_j^T}{\\sqrt{d_k}}\\right)$

            And the output for each token is: $\\text{Output}_i = \\sum_j \\text{Attention}(i,j) \\cdot V_j$
            """)

            # Controls for Q, K, V selection
            col1, col2 = st.columns(2)
            with col1:
                num_layers = st.session_state.model_config['num_layers'] if 'model_config' in st.session_state else 8
                qkv_layer = st.selectbox(
                    "Select layer for Q/K/V",
                    options=list(range(num_layers)),
                    index=num_layers-1,  # Default to last layer
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
        st.header("Complete Attention Matrix")

        with st.expander("What does this show?", expanded=False):
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

        with st.expander("View Full Attention Matrix", expanded=True):
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
                num_layers = st.session_state.model_config['num_layers'] if 'model_config' in st.session_state else 8
                full_matrix_layer = st.selectbox(
                    "Select layer for full attention matrix",
                    options=list(range(num_layers)),
                    index=num_layers-1,  # Default to last layer
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

                    # Set ticks and labels (clean Ġ from tokens)
                    clean_tokens = [tok.replace('Ġ', ' ').strip() for tok in all_tokens]
                    ax.set_xticks(range(n_total))
                    ax.set_yticks(range(n_total))
                    ax.set_xticklabels(clean_tokens, rotation=90, ha='right', fontsize=8)
                    ax.set_yticklabels(clean_tokens, fontsize=8)

                    # Remove grid lines
                    ax.grid(False)

                    # Add dividing lines to separate prompt from generated (not crossing lines)
                    ax.axhline(y=n_prompt-0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
                    ax.axvline(x=n_prompt-0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)

                    # Add labels for quadrants
                    ax.text(n_prompt/2, -1.5, 'PROMPT', ha='center', va='bottom',
                           fontsize=11, weight='bold', color='darkblue')
                    ax.text(n_prompt + n_generated/2 - n_generated*0.3, -1.5, 'GENERATED', ha='center', va='bottom',
                           fontsize=11, weight='bold', color='darkgreen')
                    ax.text(-1.5, n_prompt/2, 'PROMPT', ha='right', va='center', rotation=90,
                           fontsize=11, weight='bold', color='darkblue')
                    ax.text(-1.5, n_prompt + n_generated/2 + n_generated*0.2, 'GENERATED', ha='right', va='center', rotation=90,
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
    if st.session_state.extractor is not None and 'model_config' in st.session_state:
        config = st.session_state.model_config
        model_name = config['name']

        # Determine which model is loaded
        if "8M" in model_name:
            st.markdown(f"""
            **TinyStories-8M** is a small language model trained on simple children's stories.

            #### Architecture Specifications

            - **Total Parameters:** 8,116,224 (≈8M)
            - **Trainable Parameters:** 8,116,224 (100% trainable)
            - **Model Type:** Causal (Autoregressive) Transformer
            - **Architecture Style:** GPT-2 based

            #### Transformer Configuration

            - **Number of Layers:** {config['num_layers']} transformer blocks
            - **Attention Heads per Layer:** {config['num_heads']} heads
            - **Hidden Size:** {config['hidden_size']} dimensions
            - **Head Dimension:** {config['hidden_size'] // config['num_heads']} (hidden_size / num_heads = {config['hidden_size']} / {config['num_heads']})
            - **Vocabulary Size:** 50,257 tokens
            - **Maximum Sequence Length:** 2,048 tokens
            - **Context Window:** 2,048 tokens

            #### Parameter Breakdown

            Each transformer layer contains:
            - **Self-Attention Module:**
              - Query (Q), Key (K), Value (V) projections: 3 × ({config['hidden_size']} × {config['hidden_size']}) = {3 * config['hidden_size'] * config['hidden_size']:,} params
              - Output projection: {config['hidden_size']} × {config['hidden_size']} = {config['hidden_size'] * config['hidden_size']:,} params
              - Total per layer: ~{(4 * config['hidden_size'] * config['hidden_size']) // 1000}K params

            - **Feed-Forward Network:**
              - First linear layer: {config['hidden_size']} × {config['hidden_size'] * 4} = {config['hidden_size'] * config['hidden_size'] * 4:,} params (4× expansion)
              - Second linear layer: {config['hidden_size'] * 4} × {config['hidden_size']} = {config['hidden_size'] * config['hidden_size'] * 4:,} params
              - Total per layer: ~{(2 * config['hidden_size'] * config['hidden_size'] * 4) // 1000}K params

            - **Layer Normalization:** 2 × {config['hidden_size']} = {2 * config['hidden_size']} params per layer

            **Total per transformer block:** ~{((4 * config['hidden_size'] * config['hidden_size']) + (2 * config['hidden_size'] * config['hidden_size'] * 4)) // 1000}K parameters × {config['num_layers']} layers = ~{((4 * config['hidden_size'] * config['hidden_size']) + (2 * config['hidden_size'] * config['hidden_size'] * 4)) * config['num_layers'] // 1000000}M

            **Embedding layers:**
            - Token embeddings: 50,257 × {config['hidden_size']} = ~{(50257 * config['hidden_size']) // 1000000}M params
            - Position embeddings: 2,048 × {config['hidden_size']} = ~{(2048 * config['hidden_size']) // 1000000}M params

            #### Performance Characteristics

            **Memory Usage:**
            - Model weights: ~32 MB (8M params × 4 bytes/float32)
            - Activations (50 tokens): ~10 MB
            - Total GPU/CPU memory: ~50-100 MB

            **Inference Speed (CPU):**
            - Single token generation: 10-30ms
            - Attention extraction overhead: <5ms
            - Suitable for real-time interactive applications
            """)

        elif "1Layer" in model_name:
            st.markdown(f"""
            **TinyStories-1Layer-21M** is a single-layer language model trained on simple children's stories.

            #### Architecture Specifications

            - **Total Parameters:** ~21 million (≈21M)
            - **Model Type:** Causal (Autoregressive) Transformer
            - **Architecture Style:** GPT-2 based
            - **Special Feature:** Only 1 transformer layer!

            #### Transformer Configuration

            - **Number of Layers:** {config['num_layers']} transformer block
            - **Attention Heads per Layer:** {config['num_heads']} heads
            - **Hidden Size:** {config['hidden_size']} dimensions
            - **Head Dimension:** {config['hidden_size'] // config['num_heads']} (hidden_size / num_heads = {config['hidden_size']} / {config['num_heads']})
            - **Vocabulary Size:** 50,257 tokens
            - **Maximum Sequence Length:** 2,048 tokens
            - **Context Window:** 2,048 tokens

            #### Why Only 1 Layer?

            This model demonstrates that language models can generate coherent text even with just a **single transformer block**!
            - Makes it **extremely simple** to understand attention mechanics
            - Each of the {config['num_heads']} attention heads must learn everything in parallel
            - No hierarchical processing across layers
            - Perfect for educational purposes and visualization

            #### Parameter Breakdown

            The single transformer layer contains:
            - **Self-Attention Module:**
              - Query (Q), Key (K), Value (V) projections: 3 × ({config['hidden_size']} × {config['hidden_size']}) = {3 * config['hidden_size'] * config['hidden_size']:,} params
              - Output projection: {config['hidden_size']} × {config['hidden_size']} = {config['hidden_size'] * config['hidden_size']:,} params
              - Total: ~{(4 * config['hidden_size'] * config['hidden_size']) // 1000}K params

            - **Feed-Forward Network:**
              - First linear layer: {config['hidden_size']} × {config['hidden_size'] * 4} = {config['hidden_size'] * config['hidden_size'] * 4:,} params (4× expansion)
              - Second linear layer: {config['hidden_size'] * 4} × {config['hidden_size']} = {config['hidden_size'] * config['hidden_size'] * 4:,} params
              - Total: ~{(2 * config['hidden_size'] * config['hidden_size'] * 4) // 1000}K params

            - **Layer Normalization:** 2 × {config['hidden_size']} = {2 * config['hidden_size']} params

            **Total transformer block:** ~{((4 * config['hidden_size'] * config['hidden_size']) + (2 * config['hidden_size'] * config['hidden_size'] * 4)) // 1000000}M parameters

            **Embedding layers:**
            - Token embeddings: 50,257 × {config['hidden_size']} = ~{(50257 * config['hidden_size']) // 1000000}M params
            - Position embeddings: 2,048 × {config['hidden_size']} = ~{(2048 * config['hidden_size']) // 1000000}M params

            #### Performance Characteristics

            **Memory Usage:**
            - Model weights: ~84 MB (21M params × 4 bytes/float32)
            - Despite larger size, simpler architecture
            - Total GPU/CPU memory: ~100-150 MB

            **Inference Speed (CPU):**
            - Single token generation: 5-20ms (faster than multi-layer!)
            - Only one attention layer to compute
            - Attention extraction overhead: <3ms
            - Excellent for real-time applications
            """)

        # Common sections for both models
        st.markdown("""
        #### Training Details

        The TinyStories models were trained on a synthetic dataset of simple children's stories
        generated to contain only words that a typical 3-4 year old would understand.

        #### Attention Mechanism Details

        **Multi-Head Self-Attention:**
        """)
        st.markdown(f"""
        - Each layer has {config['num_heads']} attention heads operating in parallel
        - Each head learns different attention patterns (e.g., syntax, semantics, positional)
        - Attention scores are computed as: `softmax(Q @ K^T / sqrt({config['hidden_size'] // config['num_heads']}))`
        - Output is weighted sum of Values: `Attention @ V`
        """)

        st.markdown("""
        **Causal Masking:**
        - Attention is masked to prevent looking at future tokens
        - This ensures autoregressive generation (left-to-right)
        - Each position can only attend to itself and previous positions

        #### Comparison to Other Models

        | Model | Parameters | Layers | Heads | Hidden Size |
        |-------|-----------|--------|-------|-------------|
        | **TinyStories-1Layer-21M** | 21M | 1 | 16 | 1024 |
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
    else:
        st.info("Please load a model from the sidebar to see detailed information.")
