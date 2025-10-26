import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import plotly.graph_objects as go
import plotly.express as px
from src.model_loader import AttentionExtractor

# Page configuration
st.set_page_config(
    page_title="Embedding Explorer",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state
if 'extractor' not in st.session_state:
    st.session_state.extractor = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'vocab' not in st.session_state:
    st.session_state.vocab = None

# Main title
st.title("🔮 Embedding Space Explorer")
st.markdown("*Explore TinyStories-8M word embeddings*")

# Load model automatically on first run
if st.session_state.extractor is None:
    with st.spinner("Loading TinyStories-8M model..."):
        try:
            model_name = "roneneldan/TinyStories-8M"
            st.session_state.extractor = AttentionExtractor(model_name)

            # Extract embedding matrix and vocabulary
            embedding_matrix = st.session_state.extractor.model.get_input_embeddings().weight.detach().cpu().numpy()
            st.session_state.embeddings = embedding_matrix

            # Build vocabulary (token id -> token string)
            vocab_size = st.session_state.extractor.tokenizer.vocab_size
            st.session_state.vocab = {i: st.session_state.extractor.tokenizer.decode([i]) for i in range(vocab_size)}

            st.success(f"✅ Model loaded! Vocab: {vocab_size:,} tokens | Embedding dim: {embedding_matrix.shape[1]}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# Only show content if model is loaded
if st.session_state.extractor is not None:

    # Word selection (above tabs)
    st.subheader("Select Words to Analyze")

    col1, col2 = st.columns([1, 2])

    with col1:
        target_word = st.text_input(
            "Target word",
            value="princess",
            help="The word to compare against others"
        )

    with col2:
        comparison_words_input = st.text_input(
            "Words to compare (comma-separated)",
            value="murder, wedding",
            help="These words will be compared to the target word"
        )

    comparison_words = [w.strip() for w in comparison_words_input.split(',') if w.strip()]
    all_selected_words = [target_word] + comparison_words

    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🌐 Embedding Space",
        "🌊 Layer Evolution",
        "📊 Embedding Matrix"
    ])

    # ========================================
    # TAB 3: EMBEDDING MATRIX
    # ========================================
    with tab3:
        st.header("Embedding Matrix Heatmap")
        st.info("Visualize the full embedding matrix. All words will be labeled. Scroll to explore!")

        # Options
        col1, col2 = st.columns(2)
        with col1:
            n_words = st.number_input(
                "Number of words to show (0 = all vocabulary)",
                min_value=0,
                max_value=50000,
                value=100,
                step=50,
                help="More words = bigger plot. 0 shows everything!"
            )
        with col2:
            show_clean_only = st.checkbox("Show only clean words (no symbols)", value=True)

        if st.button("Generate Matrix Heatmap", key="matrix_heatmap_btn"):
            with st.spinner("Generating embedding matrix visualization..."):
                vocab_size = len(st.session_state.vocab)

                if show_clean_only:
                    # Filter to clean words only
                    clean_indices = []
                    clean_words = []
                    for idx in range(vocab_size):
                        word = st.session_state.vocab[idx].strip()
                        word_clean = word.replace('Ġ', '')
                        if (len(word_clean) > 0 and
                            word_clean.replace('-', '').replace("'", '').isalpha() and
                            len(word_clean) <= 15 and
                            not word.startswith('<') and not word.startswith('[') and
                            not word.startswith('|')):
                            clean_indices.append(idx)
                            clean_words.append(word_clean)
                            if n_words > 0 and len(clean_indices) >= n_words:
                                break

                    indices_to_use = clean_indices
                    words_to_use = clean_words
                else:
                    # Use all words
                    if n_words > 0:
                        indices_to_use = list(range(min(n_words, vocab_size)))
                    else:
                        indices_to_use = list(range(vocab_size))
                    words_to_use = [st.session_state.vocab[idx].strip().replace('Ġ', '') for idx in indices_to_use]

                embeddings_to_viz = st.session_state.embeddings[indices_to_use]

                st.info(f"Visualizing {len(indices_to_use)} words × {embeddings_to_viz.shape[1]} dimensions = {embeddings_to_viz.size:,} values")

                # Create interactive Plotly heatmap
                # Make height proportional to number of words (15 pixels per word for readability)
                pixels_per_word = 15
                plot_height = len(indices_to_use) * pixels_per_word

                st.warning(f"⚠️ Creating plot: {plot_height:,} pixels tall. Scroll to explore!")

                fig = go.Figure(data=go.Heatmap(
                    z=embeddings_to_viz,
                    x=[f"D{i}" for i in range(embeddings_to_viz.shape[1])],
                    y=words_to_use,
                    colorscale='Viridis',
                    hovertemplate='Word: %{y}<br>Dimension: %{x}<br>Value: %{z:.3f}<extra></extra>',
                    colorbar=dict(title="Embedding Value")
                ))

                fig.update_layout(
                    title=f"Embedding Matrix: {len(indices_to_use)} Words × {embeddings_to_viz.shape[1]} Dimensions",
                    xaxis_title="Embedding Dimension",
                    yaxis_title="Word (Vocabulary)",
                    height=plot_height,  # HUGE height so all words are readable
                    width=1400,
                    yaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=1,  # Show EVERY word
                        tickfont=dict(size=10)
                    ),
                    xaxis=dict(
                        tickfont=dict(size=10)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show some statistics
                st.subheader("Matrix Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Values", f"{embeddings_to_viz.size:,}")
                with col2:
                    st.metric("Mean", f"{embeddings_to_viz.mean():.4f}")
                with col3:
                    st.metric("Std Dev", f"{embeddings_to_viz.std():.4f}")
                with col4:
                    st.metric("Range", f"[{embeddings_to_viz.min():.2f}, {embeddings_to_viz.max():.2f}]")

    # ========================================
    # TAB 1: EMBEDDING SPACE
    # ========================================
    with tab1:
        st.header("Embedding Space Visualization")
        st.markdown("""
        Explore the embedding space after dimensionality reduction.
        Select words to highlight and analyze their similarities.
        """)

        # Dimensionality reduction method
        reduction_method = st.radio(
            "Dimensionality reduction method",
            ["PCA", "t-SNE"],
            horizontal=True,
            help="PCA is faster, t-SNE can reveal more structure"
        )

        # Precompute embedding space visualization
        cache_key = f'embedding_space_2d_{reduction_method}'
        if cache_key not in st.session_state:
            with st.spinner(f"Computing 2D embedding space with {reduction_method}..."):
                vocab_size = len(st.session_state.vocab)

                # Filter to only clean words
                clean_indices = []
                for idx in range(vocab_size):
                    word = st.session_state.vocab[idx].strip()
                    word_clean = word.replace('Ġ', '')
                    if (len(word_clean) > 0 and
                        word_clean.replace('-', '').replace("'", '').isalpha() and
                        len(word_clean) <= 15 and
                        not word.startswith('<') and not word.startswith('[') and
                        not word.startswith('|')):
                        clean_indices.append(idx)

                # Sample from clean indices
                sample_size = min(3000, len(clean_indices))
                np.random.seed(42)
                sampled_indices = np.random.choice(clean_indices, sample_size, replace=False)

                # Get embeddings for sampled words
                sampled_embeddings = st.session_state.embeddings[sampled_indices]
                sampled_words = [st.session_state.vocab[idx].strip().replace('Ġ', '') for idx in sampled_indices]

                # Apply dimensionality reduction
                if reduction_method == "PCA":
                    reducer = PCA(n_components=2)
                    reduced_embeddings = reducer.fit_transform(sampled_embeddings)
                    var_explained = reducer.explained_variance_ratio_
                else:  # t-SNE
                    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                    reduced_embeddings = reducer.fit_transform(sampled_embeddings)
                    var_explained = None

                # Store in session state
                st.session_state[cache_key] = {
                    'reduced': reduced_embeddings,
                    'words': sampled_words,
                    'indices': sampled_indices,
                    'reducer': reducer,
                    'var_explained': var_explained
                }
                st.success(f"✓ Embedding space computed with {reduction_method}! ({sample_size} clean words)")

        # Get cached data
        space_data = st.session_state[cache_key]

        if st.button("Visualize", key="viz_space_btn"):
            try:
                # Get embeddings and positions for all words
                selected_embeddings = []
                selected_positions = []
                selected_words = []

                for word in all_selected_words:
                    try:
                        token_id = st.session_state.extractor.tokenizer.encode(word, add_special_tokens=False)[0]
                        emb = st.session_state.embeddings[token_id]

                        # Project to 2D
                        if reduction_method == "PCA":
                            pos_2d = space_data['reducer'].transform([emb])[0]
                        else:
                            # For t-SNE, find nearest neighbor in the already reduced space
                            distances = np.linalg.norm(st.session_state.embeddings[space_data['indices']] - emb, axis=1)
                            nearest_idx = np.argmin(distances)
                            pos_2d = space_data['reduced'][nearest_idx]

                        selected_embeddings.append(emb)
                        selected_positions.append(pos_2d)
                        selected_words.append(word)
                    except:
                        st.warning(f"Could not find word: {word}")

                if len(selected_words) < 2:
                    st.error("Need at least target word + 1 comparison word")
                    raise ValueError("Not enough words")

                # Create Plotly visualization
                fig = go.Figure()

                # Add background vocabulary
                bg_x = space_data['reduced'][:, 0]
                bg_y = space_data['reduced'][:, 1]
                bg_words = space_data['words']

                fig.add_trace(go.Scatter(
                    x=bg_x,
                    y=bg_y,
                    mode='markers',
                    marker=dict(size=4, color='lightgray', opacity=0.4),
                    text=bg_words,
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    name='Vocabulary',
                    showlegend=True
                ))

                # Add selected words
                if len(selected_positions) > 0:
                    selected_positions = np.array(selected_positions)

                    # Target word (first one)
                    fig.add_trace(go.Scatter(
                        x=[selected_positions[0, 0]],
                        y=[selected_positions[0, 1]],
                        mode='markers+text',
                        marker=dict(size=20, color='green', symbol='star', line=dict(width=2, color='white')),
                        text=[selected_words[0]],
                        textposition='top center',
                        textfont=dict(size=14, color='green', family='Arial Black'),
                        hovertemplate='<b>TARGET: %{text}</b><extra></extra>',
                        name=f'Target: {selected_words[0]}',
                        showlegend=True
                    ))

                    # Comparison words (rest)
                    if len(selected_positions) > 1:
                        fig.add_trace(go.Scatter(
                            x=selected_positions[1:, 0],
                            y=selected_positions[1:, 1],
                            mode='markers+text',
                            marker=dict(size=15, color='orange', symbol='diamond', line=dict(width=2, color='white')),
                            text=selected_words[1:],
                            textposition='top center',
                            textfont=dict(size=12, color='orange', family='Arial Black'),
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            name='Comparison words',
                            showlegend=True
                        ))

                # Update layout
                if reduction_method == "PCA" and space_data['var_explained'] is not None:
                    xaxis_title = f"PC1 ({space_data['var_explained'][0]:.1%} variance)"
                    yaxis_title = f"PC2 ({space_data['var_explained'][1]:.1%} variance)"
                else:
                    xaxis_title = "Dimension 1"
                    yaxis_title = "Dimension 2"

                fig.update_layout(
                    title=f"Embedding Space ({reduction_method})",
                    xaxis_title=xaxis_title,
                    yaxis_title=yaxis_title,
                    height=700,
                    hovermode='closest',
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Similarity analysis
                if len(selected_embeddings) > 1:
                    st.subheader("Similarity Analysis: Which word is nearest to the target?")
                    st.caption(f"Target word: **{selected_words[0]}** | Comparison words: {', '.join(selected_words[1:])}")

                    # Compute ALL metrics
                    results = {}
                    selected_embeddings_array = np.array(selected_embeddings)
                    from scipy.spatial.distance import cdist

                    # Cosine Similarity
                    results["Cosine Similarity"] = {
                        'data': cosine_similarity(selected_embeddings),
                        'cmap': 'YlOrRd',
                        'vmin': 0,
                        'vmax': 1,
                        'note': 'higher = more similar',
                        'format': '.3f',
                        'higher_is_closer': True
                    }

                    # Euclidean Distance
                    results["Euclidean Distance"] = {
                        'data': euclidean_distances(selected_embeddings),
                        'cmap': 'YlGnBu',
                        'vmin': None,
                        'vmax': None,
                        'note': 'lower = more similar',
                        'format': '.1f',
                        'higher_is_closer': False
                    }

                    # Manhattan Distance
                    manhattan = cdist(selected_embeddings_array, selected_embeddings_array, metric='cityblock')
                    results["Manhattan Distance"] = {
                        'data': manhattan,
                        'cmap': 'Purples',
                        'vmin': None,
                        'vmax': None,
                        'note': 'lower = more similar',
                        'format': '.1f',
                        'higher_is_closer': False
                    }

                    # Dot Product
                    dot_prod = selected_embeddings_array @ selected_embeddings_array.T
                    results["Dot Product"] = {
                        'data': dot_prod,
                        'cmap': 'RdYlGn',
                        'vmin': None,
                        'vmax': None,
                        'note': 'higher = more similar',
                        'format': '.1f',
                        'higher_is_closer': True
                    }

                    # Chebyshev Distance
                    chebyshev = cdist(selected_embeddings_array, selected_embeddings_array, metric='chebyshev')
                    results["Chebyshev Distance"] = {
                        'data': chebyshev,
                        'cmap': 'Oranges',
                        'vmin': None,
                        'vmax': None,
                        'note': 'lower = more similar',
                        'format': '.2f',
                        'higher_is_closer': False
                    }

                    # Define explanations for each metric
                    explanations = {
                            "Cosine Similarity": {
                                "desc": "Measures the cosine of the angle between vectors. Range [0, 1] for positive embeddings.",
                                "note": "1 = identical direction\n0 = orthogonal\n-1 = opposite",
                                "equation": r"\text{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}"
                            },
                            "Euclidean Distance": {
                                "desc": "L2 norm - straight-line distance between points. Range [0, ∞).",
                                "note": "Most common distance metric",
                                "equation": r"d_{\text{Euclidean}}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}"
                            },
                            "Manhattan Distance": {
                                "desc": "L1 norm - sum of absolute differences. Range [0, ∞).",
                                "note": "Also called 'taxicab' or 'city block' distance",
                                "equation": r"d_{\text{Manhattan}}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_1 = \sum_{i=1}^{n} |u_i - v_i|"
                            },
                            "Dot Product": {
                                "desc": "Raw inner product without normalization. Range (-∞, ∞).",
                                "note": "Similar to cosine but magnitude-dependent",
                                "equation": r"\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i"
                            },
                            "Chebyshev Distance": {
                                "desc": "L∞ norm - maximum difference along any single dimension. Range [0, ∞).",
                                "note": "Also called 'infinity norm'",
                                "equation": r"d_{\text{Chebyshev}}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_\infty = \max_{i=1}^{n} |u_i - v_i|"
                            }
                    }

                    # Info note at the top
                    st.info(f"⚠️ All metrics calculated in FULL {st.session_state.embeddings.shape[1]}-dimensional space (not the 2D projection)")

                    # Create two-column layout for each metric
                    for metric_name, metric_info in results.items():
                        data = metric_info['data']

                        # Find nearest word to target (row 0, excluding column 0 which is target itself)
                        target_row = data[0, 1:]  # Target's distances/similarities to comparison words
                        if metric_info['higher_is_closer']:
                            # For similarity metrics (higher = closer)
                            nearest_idx = np.argmax(target_row) + 1  # +1 because we excluded column 0
                        else:
                            # For distance metrics (lower = closer)
                            nearest_idx = np.argmin(target_row) + 1  # +1 because we excluded column 0

                        nearest_word = selected_words[nearest_idx]

                        col_plot, col_explain = st.columns([1.2, 1])

                        # Left column: Plot
                        with col_plot:
                            fig_single, ax = plt.subplots(figsize=(7, 5.5))

                            im = ax.imshow(data, cmap=metric_info['cmap'],
                                         vmin=metric_info['vmin'], vmax=metric_info['vmax'])
                            ax.set_xticks(range(len(selected_words)))
                            ax.set_yticks(range(len(selected_words)))
                            ax.set_xticklabels(selected_words, rotation=45, ha='right')
                            ax.set_yticklabels(selected_words)
                            ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold')
                            plt.colorbar(im, ax=ax)

                            # Annotate cells with values
                            for i in range(len(selected_words)):
                                for j in range(len(selected_words)):
                                    # Highlight the nearest word to target
                                    if i == 0 and j == nearest_idx:
                                        # This is the nearest word!
                                        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=3)
                                        fontweight = 'bold'
                                        fontsize = 11
                                    else:
                                        bbox_props = None
                                        fontweight = 'normal'
                                        fontsize = 10

                                    ax.text(j, i, format(data[i, j], metric_info['format']),
                                           ha="center", va="center", color="black",
                                           fontsize=fontsize, fontweight=fontweight,
                                           bbox=bbox_props)

                            plt.tight_layout()
                            st.pyplot(fig_single)

                            # Show which word is nearest
                            st.success(f"🎯 Nearest to **{selected_words[0]}**: **{nearest_word}**")

                        # Right column: Explanation
                        with col_explain:
                            if metric_name in explanations:
                                exp = explanations[metric_name]
                                st.markdown(f"### {metric_name}")
                                st.markdown(f"**{exp['desc']}**")
                                st.markdown(f"*{exp['note']}*")
                                st.markdown("**Formula:**")
                                st.latex(exp['equation'])
                                st.caption(f"where n = {st.session_state.embeddings.shape[1]}")

                        st.markdown("---")  # Separator between metrics

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # ========================================
    # TAB 2: LAYER EVOLUTION
    # ========================================
    with tab2:
        st.header("Layer Evolution: How Distances Change Through the Model")
        st.markdown("""
        Write a prompt containing your selected words and see how their similarities evolve
        as the prompt passes through each transformer layer.
        """)

        prompt_text = st.text_area(
            "Enter a prompt (should contain your selected words)",
            value="The princess went to the wedding but saw a murder.",
            height=100,
            help="The prompt should ideally contain the words you selected above"
        )

        if st.button("Analyze Layer Evolution", key="layer_evolution_btn"):
            with st.spinner("Extracting representations from all layers..."):
                try:
                    # Extract activations from all layers
                    result = st.session_state.extractor.extract_activations(prompt_text)
                    tokens = result['tokens']
                    hidden_states = result['hidden_states']
                    n_layers = len(hidden_states) - 1  # Exclude embedding layer

                    st.success(f"✓ Extracted {n_layers} layers!")

                    # Find token positions for selected words
                    word_positions = {}
                    for word in all_selected_words:
                        # Clean the word for comparison
                        word_clean = word.strip().lower()

                        # Search through tokens by string matching
                        for i, token in enumerate(tokens):
                            # Remove GPT-2 space marker and clean
                            token_clean = token.replace('Ġ', '').strip().lower()

                            # Check if the word matches the token
                            if word_clean == token_clean or word_clean in token_clean or token_clean in word_clean:
                                word_positions[word] = i
                                break

                    if len(word_positions) < len(all_selected_words):
                        missing = set(all_selected_words) - set(word_positions.keys())
                        st.warning(f"⚠️ Could not find these words in the prompt: {', '.join(missing)}")
                        st.info("The selected words should appear in the prompt for best results.")

                    if len(word_positions) >= 2:
                        st.subheader(f"Found words: {', '.join(word_positions.keys())}")
                        st.caption(f"Prompt tokens: {' '.join(tokens)}")

                        # Calculate all metrics across all layers
                        from scipy.spatial.distance import cdist

                        layers_data = {}
                        for layer_idx in range(n_layers + 1):  # +1 to include embedding layer
                            # Extract embeddings for this layer
                            layer_embeddings = []
                            layer_words = []

                            for word, pos in word_positions.items():
                                emb = hidden_states[layer_idx][0, pos, :].cpu().numpy()
                                layer_embeddings.append(emb)
                                layer_words.append(word)

                            if len(layer_embeddings) >= 2:
                                emb_array = np.array(layer_embeddings)

                                # Calculate all metrics
                                cos_sim = cosine_similarity(emb_array)
                                euc_dist = euclidean_distances(emb_array)
                                manhattan = cdist(emb_array, emb_array, metric='cityblock')
                                dot_prod = emb_array @ emb_array.T
                                chebyshev = cdist(emb_array, emb_array, metric='chebyshev')

                                layers_data[layer_idx] = {
                                    'words': layer_words,
                                    'Cosine Similarity': cos_sim,
                                    'Euclidean Distance': euc_dist,
                                    'Manhattan Distance': manhattan,
                                    'Dot Product': dot_prod,
                                    'Chebyshev Distance': chebyshev
                                }

                        # Plot evolution for each metric
                        st.info(f"⚠️ Tracking similarities in FULL {st.session_state.embeddings.shape[1]}-D hidden states across {n_layers + 1} layers")

                        # For each metric, show how distance between target and each comparison word evolves
                        target_word_idx = 0  # First word is target

                        metrics_info = {
                            'Cosine Similarity': {'higher_is_closer': True, 'color': 'green'},
                            'Euclidean Distance': {'higher_is_closer': False, 'color': 'blue'},
                            'Manhattan Distance': {'higher_is_closer': False, 'color': 'purple'},
                            'Dot Product': {'higher_is_closer': True, 'color': 'red'},
                            'Chebyshev Distance': {'higher_is_closer': False, 'color': 'orange'}
                        }

                        for metric_name, metric_settings in metrics_info.items():
                            st.subheader(f"{metric_name} Evolution")

                            fig, ax = plt.subplots(figsize=(12, 6))

                            # For each comparison word, plot its distance to target across layers
                            for comp_idx in range(1, len(layer_words)):
                                comp_word = layer_words[comp_idx]
                                values = []

                                for layer_idx in sorted(layers_data.keys()):
                                    data_matrix = layers_data[layer_idx][metric_name]
                                    value = data_matrix[target_word_idx, comp_idx]
                                    values.append(value)

                                ax.plot(sorted(layers_data.keys()), values, marker='o',
                                       label=f'{layer_words[0]} ↔ {comp_word}', linewidth=2, markersize=8)

                            ax.set_xlabel('Layer', fontsize=12)
                            ax.set_ylabel(metric_name, fontsize=12)
                            ax.set_title(f'How {metric_name} Changes Across Layers', fontsize=14, fontweight='bold')
                            ax.legend(loc='best')
                            ax.grid(True, alpha=0.3)
                            ax.set_xticks(range(n_layers + 1))
                            ax.set_xticklabels(['Emb'] + [f'L{i}' for i in range(1, n_layers + 1)])

                            # Add annotation for what the metric means
                            if metric_settings['higher_is_closer']:
                                ax.text(0.02, 0.98, '↑ Higher = More Similar',
                                       transform=ax.transAxes, fontsize=10,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                            else:
                                ax.text(0.02, 0.98, '↓ Lower = More Similar',
                                       transform=ax.transAxes, fontsize=10,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                            plt.tight_layout()
                            st.pyplot(fig)

                    else:
                        st.error("Need at least 2 words found in the prompt to analyze!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

else:
    st.info("👈 Loading model...")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • TinyStories-8M • Explore the geometry of language*")
