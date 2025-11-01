import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import plotly.graph_objects as go
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
st.markdown("...Once upon a time, finally see 256-dimensional embeddings and lose your words.")

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

    # Show embedding vectors for selected words
    selected_embeddings = []
    if len(all_selected_words) > 0:
        st.subheader("Embedding Vectors")
        try:
            for word in all_selected_words:
                token_id = st.session_state.extractor.tokenizer.encode(word, add_special_tokens=False)[0]
                emb = st.session_state.embeddings[token_id]
                selected_embeddings.append(emb)

                is_target = word == target_word
                if is_target:
                    st.markdown(f"**🎯 {word}** (target) - 𝐮:")
                else:
                    st.markdown(f"**{word}** - 𝐯:")

                st.code(f"[{', '.join([f'{v:.3f}' for v in emb])}]", language=None)
        except Exception as e:
            st.warning(f"Could not load embeddings: {e}")

    # --- Similarity Analysis (doesn't depend on PCA) ---
    if len(selected_embeddings) > 1 and len(all_selected_words) > 1:
        st.markdown("---")
        st.subheader("Similarity Analysis: Which word is nearest to the target?")
        st.caption(
            f"Target word: **{all_selected_words[0]}** | Comparison words: {', '.join(all_selected_words[1:])}")

        from scipy.spatial.distance import cdist

        # --- Compute metrics ---
        selected_embeddings_array = np.array(selected_embeddings)
        results = {
            "Cosine Similarity": {
                'data': cosine_similarity(selected_embeddings),
                'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 1,
                'note': 'higher = more similar', 'format': '.3f',
                'higher_is_closer': True
            },
            "Euclidean Distance": {
                'data': euclidean_distances(selected_embeddings),
                'cmap': 'YlGnBu', 'note': 'lower = more similar',
                'format': '.1f', 'higher_is_closer': False
            },
            "Manhattan Distance": {
                'data': cdist(selected_embeddings_array, selected_embeddings_array, metric='cityblock'),
                'cmap': 'Purples', 'note': 'lower = more similar',
                'format': '.1f', 'higher_is_closer': False
            },
            "Dot Product": {
                'data': selected_embeddings_array @ selected_embeddings_array.T,
                'cmap': 'RdYlGn', 'note': 'higher = more similar',
                'format': '.1f', 'higher_is_closer': True
            },
            "Chebyshev Distance": {
                'data': cdist(selected_embeddings_array, selected_embeddings_array, metric='chebyshev'),
                'cmap': 'Oranges', 'note': 'lower = more similar',
                'format': '.2f', 'higher_is_closer': False
            }
        }

        explanations = {
            "Cosine Similarity": {
                "desc": "Measures the cosine of the angle between vectors. Range [0, 1] for positive embeddings.",
                "note": "1 = identical direction, 0 = orthogonal, -1 = opposite",
                "equation": r"\text{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}"
            },
            "Euclidean Distance": {
                "desc": "L2 norm - straight-line distance between points.",
                "note": "Most common distance metric",
                "equation": r"d_{\text{Euc}}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_i - v_i)^2}"
            },
            "Manhattan Distance": {
                "desc": "L1 norm - sum of absolute differences.",
                "note": "Also called 'taxicab' distance",
                "equation": r"d_{\text{Man}}(\mathbf{u}, \mathbf{v}) = \sum |u_i - v_i|"
            },
            "Dot Product": {
                "desc": "Raw inner product without normalization.",
                "note": "Similar to cosine but magnitude-dependent",
                "equation": r"\mathbf{u}\cdot\mathbf{v} = \sum u_i v_i"
            },
            "Chebyshev Distance": {
                "desc": "L∞ norm - maximum difference along any single dimension.",
                "note": "Also called 'infinity norm'",
                "equation": r"d_{\text{Cheb}}(\mathbf{u}, \mathbf{v}) = \max |u_i - v_i|"
            }
        }

        # --- Show results horizontally with explanations underneath ---
        num_cols = 5  # number of plots per row
        metrics_list = list(results.items())

        for row_start in range(0, len(metrics_list), num_cols):
            cols = st.columns(num_cols)

            for i, (metric_name, metric_info) in enumerate(metrics_list[row_start: row_start + num_cols]):
                with cols[i]:
                    data = metric_info['data']
                    target_row = data[0, 1:]
                    if metric_info['higher_is_closer']:
                        nearest_idx = np.argmax(target_row) + 1
                    else:
                        nearest_idx = np.argmin(target_row) + 1
                    nearest_word = all_selected_words[nearest_idx]

                    # --- Plot ---
                    fig_single, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(data, cmap=metric_info['cmap'],
                                   vmin=metric_info.get('vmin'),
                                   vmax=metric_info.get('vmax'))
                    ax.set_xticks(range(len(all_selected_words)))
                    ax.set_yticks(range(len(all_selected_words)))
                    ax.set_xticklabels(all_selected_words, rotation=45, ha='right')
                    ax.set_yticklabels(all_selected_words)
                    ax.set_title(metric_name, fontsize=12, fontweight='bold')
                    plt.colorbar(im, ax=ax)

                    # Annotate
                    for r in range(len(all_selected_words)):
                        for c in range(len(all_selected_words)):
                            val = format(data[r, c], metric_info['format'])
                            ax.text(c, r, val, ha='center', va='center', color='black', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig_single)
                    st.success(f"🎯 Nearest to **{all_selected_words[0]}**: **{nearest_word}**")

                    # --- Explanation below each plot ---
                    if metric_name in explanations:
                        exp = explanations[metric_name]
                        st.markdown(f"**{exp['desc']}**")
                        st.caption(f"*{exp['note']}*")
                        st.latex(exp['equation'])

                        # --- Show numerical calculation example ---
                        with st.expander("🔢 See full calculation with numbers"):
                            # Calculate for target vs nearest comparison word
                            target_vec = selected_embeddings_array[0]
                            comp_vec = selected_embeddings_array[nearest_idx]

                            st.markdown(f"**Example: {all_selected_words[0]} vs {nearest_word}**")
                            st.markdown(f"**Variable mapping:**")
                            st.latex(rf"\mathbf{{u}} = \text{{{all_selected_words[0]}}}, \quad \mathbf{{v}} = \text{{{nearest_word}}}")

                            st.caption(f"Total embedding dimension: {len(target_vec)}")

                            # Always show all dimensions
                            n_show = len(target_vec)

                            # Show vectors
                            st.markdown(f"**All {n_show} dimensions:**")
                            st.code(f"𝐮 ({all_selected_words[0]}): [{', '.join([f'{v:.3f}' for v in target_vec])}]")
                            st.code(f"𝐯 ({nearest_word}): [{', '.join([f'{v:.3f}' for v in comp_vec])}]")

                            if metric_name == "Cosine Similarity":
                                dot_prod = np.dot(target_vec, comp_vec)
                                norm_target = np.linalg.norm(target_vec)
                                norm_comp = np.linalg.norm(comp_vec)
                                result = dot_prod / (norm_target * norm_comp)

                                st.markdown("**Calculation:**")
                                st.latex(rf"\mathbf{{u}} \cdot \mathbf{{v}} = \sum_{{i=1}}^{{{len(target_vec)}}} u_i v_i = {dot_prod:.6f}")
                                st.latex(rf"\|\mathbf{{u}}\| = \sqrt{{\sum_{{i=1}}^{{{len(target_vec)}}} u_i^2}} = {norm_target:.6f}")
                                st.latex(rf"\|\mathbf{{v}}\| = \sqrt{{\sum_{{i=1}}^{{{len(target_vec)}}} v_i^2}} = {norm_comp:.6f}")
                                st.latex(rf"\text{{cosine}}(\mathbf{{u}}, \mathbf{{v}}) = \frac{{{dot_prod:.6f}}}{{{norm_target:.6f} \times {norm_comp:.6f}}} = {result:.6f}")

                            elif metric_name == "Euclidean Distance":
                                diff = target_vec - comp_vec
                                squared_diff = diff ** 2
                                sum_squared = np.sum(squared_diff)
                                result = np.sqrt(sum_squared)

                                st.markdown("**Calculation:**")
                                st.latex(rf"\sum_{{i=1}}^{{{len(target_vec)}}} (u_i - v_i)^2 = {sum_squared:.6f}")
                                st.latex(rf"d_{{Euc}} = \sqrt{{{sum_squared:.6f}}} = {result:.6f}")

                            elif metric_name == "Manhattan Distance":
                                diff = np.abs(target_vec - comp_vec)
                                result = np.sum(diff)

                                st.markdown("**Calculation:**")
                                st.latex(rf"d_{{Man}} = \sum_{{i=1}}^{{{len(target_vec)}}} |u_i - v_i| = {result:.6f}")

                            elif metric_name == "Dot Product":
                                products = target_vec * comp_vec
                                result = np.sum(products)

                                st.markdown("**Calculation:**")
                                st.latex(rf"\mathbf{{u}} \cdot \mathbf{{v}} = \sum_{{i=1}}^{{{len(target_vec)}}} u_i v_i = {result:.6f}")

                            elif metric_name == "Chebyshev Distance":
                                diff = np.abs(target_vec - comp_vec)
                                result = np.max(diff)
                                max_idx = np.argmax(diff)

                                st.markdown("**Calculation:**")
                                st.latex(rf"d_{{Cheb}} = \max_{{i}} |u_i - v_i| = |u_{{{max_idx}}} - v_{{{max_idx}}}| = {result:.6f}")
                                st.caption(f"Maximum difference at dimension {max_idx}: |{target_vec[max_idx]:.3f} - {comp_vec[max_idx]:.3f}| = {result:.6f}")

                            st.success(f"✅ Final result: **{data[0, nearest_idx]:.6f}**")

    st.markdown("---")

    # Unified word range slider for both tabs
    st.subheader("Select Vocabulary Range")

    # Checkbox for clean vs all vocabulary
    use_clean_vocab = st.checkbox(
        "Use clean vocabulary only (alphabetic words, no special tokens)",
        value=True,
        help="When checked, only shows alphabetic words without special tokens. Uncheck to see all tokens including special characters."
    )

    # Build vocabulary list based on checkbox
    vocab_size = len(st.session_state.vocab)

    if use_clean_vocab:
        words_list = []
        for idx in range(vocab_size):
            word = st.session_state.vocab[idx].strip()
            word_clean = word.replace('Ġ', '')
            if (len(word_clean) > 0 and
                    word_clean.replace('-', '').replace("'", '').isalpha() and
                    len(word_clean) <= 15 and
                    not word.startswith('<') and not word.startswith('[') and
                    not word.startswith('|')):
                words_list.append(word_clean)
    else:
        # All vocabulary (raw tokens)
        words_list = [st.session_state.vocab[idx].strip().replace('Ġ', '') for idx in range(vocab_size)]

    total_words = len(words_list)
    vocab_type = "clean vocabulary" if use_clean_vocab else "all tokens"

    start_idx, end_idx = st.slider(
        f"Word range (from {vocab_type})",
        min_value=0,
        max_value=max(1, total_words - 1),
        value=(0, min(1000, total_words - 1)),  # Default to 1000 words
        step=1,
        help=f"Select which part of the {vocab_type} to use for visualizations"
    )

    # Store for use in tabs
    clean_words_all = words_list

    st.markdown("---")

    # Create tabs
    tab1, tab3 = st.tabs([
        "🌐 Embedding Space",
        "📊 Embedding Matrix"
    ])

    # ========================================
    # TAB 3: EMBEDDING MATRIX
    # ========================================
    with tab3:
        st.header("Embedding Matrix Heatmap")
        st.info(f"Showing words {start_idx:,} to {end_idx:,} from the vocabulary range selected above.")

        with st.spinner("Generating embedding matrix visualization..."):
                # --- Get selected words embeddings ---
                selected_words_info = []
                for word in all_selected_words:
                    try:
                        token_id = st.session_state.extractor.tokenizer.encode(word, add_special_tokens=False)[0]
                        emb = st.session_state.embeddings[token_id]
                        selected_words_info.append({
                            'word': word,
                            'token_id': token_id,
                            'embedding': emb,
                            'is_target': word == target_word
                        })
                    except:
                        st.warning(f"Could not find word: {word}")

                # Apply the range slice for background words
                words_to_use = clean_words_all[start_idx:end_idx + 1]
                # Get actual token indices for these words
                indices_to_use = []
                for word in words_to_use:
                    for idx, vocab_word in st.session_state.vocab.items():
                        if vocab_word.strip().replace('Ġ', '') == word:
                            indices_to_use.append(idx)
                            break
                embeddings_background = st.session_state.embeddings[indices_to_use, :]

                # Combine selected words (at bottom) with background range
                if len(selected_words_info) > 0:
                    selected_embeddings = np.array([info['embedding'] for info in selected_words_info])
                    # Make selected words bold
                    selected_word_labels = [f"<b>{info['word']}</b>" for info in selected_words_info]

                    # Stack: background words first, then selected words at bottom
                    embeddings_to_viz = np.vstack([embeddings_background, selected_embeddings])
                    y_labels = words_to_use + selected_word_labels
                else:
                    embeddings_to_viz = embeddings_background
                    y_labels = words_to_use

                n_selected = len(selected_words_info)
                st.info(
                    f"Showing **{n_selected} selected words** (highlighted) + words {start_idx}–{end_idx} ({len(words_to_use)} words) × {embeddings_to_viz.shape[1]} dimensions = {embeddings_to_viz.size:,} values"
                )

                # --- Plot setup ---
                # Use larger pixels per word to ensure visibility
                pixels_per_word = 20
                plot_height = len(y_labels) * pixels_per_word
                st.warning(f"⚠️ Creating plot: {plot_height:,} pixels tall. Scroll to explore!")

                # Use percentile-based color scaling for better contrast
                vmin, vmax = np.percentile(embeddings_to_viz, [5, 95])

                # Use Viridis colorscale
                fig = go.Figure(data=go.Heatmap(
                    z=embeddings_to_viz,
                    x=[f"D{i}" for i in range(embeddings_to_viz.shape[1])],
                    y=y_labels,
                    colorscale='Viridis',
                    zmin=vmin,
                    zmax=vmax,
                    hovertemplate='Word: %{y}<br>Dimension: %{x}<br>Value: %{z:.3f}<extra></extra>',
                    colorbar=dict(title="Embedding Value"),
                    xgap=0,  # No gap between cells
                    ygap=0   # No gap between cells
                ))

                fig.update_layout(
                    title=f"Embedding Matrix: {n_selected} Selected Words + Background Range {start_idx}–{end_idx}",
                    xaxis_title="Embedding Dimension",
                    yaxis_title="Word (Vocabulary)",
                    height=plot_height,
                    width=1400,
                    yaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=1,
                        tickfont=dict(size=10)  # Remove color='black' for dark mode compatibility
                    ),
                    xaxis=dict(
                        tickfont=dict(size=10)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- Statistics ---
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

        # Expander for PCA explanation
        with st.expander("📖 How PCA Works - Complete Explanation"):
            st.markdown("### Principal Component Analysis (PCA)")

            st.markdown("**Goal:** Find the directions (principal components) of maximum variance in high-dimensional data")

            st.markdown("---")
            st.markdown("#### Step 1: Center the Data")
            st.latex(r"\bar{\mathbf{X}} = \mathbf{X} - \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i")
            st.markdown(r"Subtract the mean to center data at the origin")

            st.markdown("#### Step 2: Compute Covariance Matrix")
            st.latex(r"\mathbf{\Sigma} = \frac{1}{n-1} \bar{\mathbf{X}}^T \bar{\mathbf{X}}")
            st.markdown(r"where $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$ captures how dimensions vary together")

            st.markdown("#### Step 3: Eigendecomposition")
            st.latex(r"\mathbf{\Sigma} \mathbf{w}_i = \lambda_i \mathbf{w}_i")
            st.markdown(r"Find eigenvectors $\mathbf{w}_i$ (principal components) and eigenvalues $\lambda_i$ (variance explained)")

            st.markdown("#### Step 4: Project to 2D")
            st.latex(r"\mathbf{Z} = \bar{\mathbf{X}} \mathbf{W}")
            st.markdown(r"where $\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2] \in \mathbb{R}^{d \times 2}$ contains the top 2 eigenvectors, and $\mathbf{Z} \in \mathbb{R}^{n \times 2}$ are the 2D coordinates")

            st.markdown("---")
            st.markdown("#### Variance Explained")
            st.markdown(r"The fraction of total variance captured by PC$i$:")
            st.latex(r"\text{Var. Explained}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}")
            st.markdown("Higher percentages mean the 2D visualization preserves more information from the original high-dimensional space")

            st.markdown("---")
            st.markdown("#### Key Properties")
            st.markdown("""
            - **Linear transformation:** Can project new points without refitting
            - **Orthogonal components:** PC1 ⊥ PC2 (uncorrelated)
            - **Deterministic:** Same data → same result every time
            - **Global structure:** Preserves large-scale relationships
            - **Optimal reconstruction:** Minimizes squared reconstruction error
            """)

        # Use PCA only
        reduction_method = "PCA"

        # Cache key depends on slider range and vocabulary type (not selected words)
        vocab_cache_key = "clean" if use_clean_vocab else "all"
        cache_key = f'embedding_space_2d_{reduction_method}_{vocab_cache_key}_{start_idx}_{end_idx}'

        if cache_key not in st.session_state:
            with st.spinner(f"Computing PCA on vocabulary range..."):
                # Get words from the slider range
                words_in_range = clean_words_all[start_idx:end_idx + 1]

                # Get token indices for these words
                range_indices = []
                for word in words_in_range:
                    for idx, vocab_word in st.session_state.vocab.items():
                        if vocab_word.strip().replace('Ġ', '') == word:
                            range_indices.append(idx)
                            break

                # Fit PCA only on background vocabulary
                sampled_embeddings = st.session_state.embeddings[range_indices]
                sampled_words = words_in_range

                # Apply PCA
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(sampled_embeddings)
                var_explained = reducer.explained_variance_ratio_

                # Store in session state
                st.session_state[cache_key] = {
                    'reduced': reduced_embeddings,
                    'words': sampled_words,
                    'indices': range_indices,
                    'reducer': reducer,
                    'var_explained': var_explained
                }

                st.success(f"✓ PCA fitted on {len(range_indices)} vocabulary words!")

        # Get cached data
        space_data = st.session_state[cache_key]

        try:
                # --- Get embeddings and positions for all words ---
                selected_embeddings, selected_positions, selected_words = [], [], []

                for word_idx, word in enumerate(all_selected_words):
                    try:
                        token_id = st.session_state.extractor.tokenizer.encode(word, add_special_tokens=False)[0]
                        emb = st.session_state.embeddings[token_id]

                        # Check if word is in background range
                        if token_id in space_data['indices']:
                            # Use pre-computed position
                            pos_in_background = space_data['indices'].index(token_id)
                            pos_2d = space_data['reduced'][pos_in_background]
                        else:
                            # Transform using fitted PCA
                            pos_2d = space_data['reducer'].transform([emb])[0]

                        selected_embeddings.append(emb)
                        selected_positions.append(pos_2d)
                        selected_words.append(word)
                    except Exception as e:
                        st.warning(f"Could not find word: {word} ({e})")

                if len(selected_words) < 2:
                    st.error("Need at least target word + 1 comparison word")
                    raise ValueError("Not enough words")

                # --- 2D Embedding Plot ---
                fig = go.Figure()

                # Background
                bg_x = space_data['reduced'][:, 0]
                bg_y = space_data['reduced'][:, 1]
                bg_words = space_data['words']
                fig.add_trace(go.Scatter(
                    x=bg_x, y=bg_y, mode='markers',
                    marker=dict(size=4, color='lightgray', opacity=0.4),
                    text=bg_words, hovertemplate='<b>%{text}</b><extra></extra>',
                    name='Vocabulary', showlegend=True
                ))

                # Selected words
                if len(selected_positions) > 0:
                    selected_positions = np.array(selected_positions)

                    # Target word
                    fig.add_trace(go.Scatter(
                        x=[selected_positions[0, 0]], y=[selected_positions[0, 1]],
                        mode='markers+text',
                        marker=dict(size=20, color='green', symbol='star',
                                    line=dict(width=2, color='white')),
                        text=[selected_words[0]], textposition='top center',
                        textfont=dict(size=14, color='green', family='Arial Black'),
                        hovertemplate='<b>TARGET: %{text}</b><extra></extra>',
                        name=f'Target: {selected_words[0]}'
                    ))

                    # Comparison words
                    if len(selected_positions) > 1:
                        fig.add_trace(go.Scatter(
                            x=selected_positions[1:, 0], y=selected_positions[1:, 1],
                            mode='markers+text',
                            marker=dict(size=15, color='orange', symbol='diamond',
                                        line=dict(width=2, color='white')),
                            text=selected_words[1:], textposition='top center',
                            textfont=dict(size=12, color='orange', family='Arial Black'),
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            name='Comparison words'
                        ))

                # --- Layout ---
                if reduction_method == "PCA" and space_data['var_explained'] is not None:
                    xaxis_title = f"PC1 ({space_data['var_explained'][0]:.1%} variance)"
                    yaxis_title = f"PC2 ({space_data['var_explained'][1]:.1%} variance)"
                else:
                    xaxis_title, yaxis_title = "Dimension 1", "Dimension 2"

                fig.update_layout(
                    title=f"Embedding Space ({reduction_method})",
                    xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                    height=700, hovermode='closest',
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False)
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- Show actual PCA calculations with numbers ---
                with st.expander("🔢 Actual Fitted PCA Components & Example Calculation"):
                    pc1 = space_data['reducer'].components_[0]
                    pc2 = space_data['reducer'].components_[1]
                    eigenvalues = space_data['reducer'].explained_variance_
                    mean_vec = space_data['reducer'].mean_

                    st.markdown(f"**Principal Component 1 (PC1)** - captures {space_data['var_explained'][0]:.2%} of variance")
                    st.caption(f"Eigenvalue λ₁ = {eigenvalues[0]:.6f}")
                    st.code(f"PC1 = [{', '.join([f'{v:.3f}' for v in pc1])}]")

                    st.markdown(f"**Principal Component 2 (PC2)** - captures {space_data['var_explained'][1]:.2%} of variance")
                    st.caption(f"Eigenvalue λ₂ = {eigenvalues[1]:.6f}")
                    st.code(f"PC2 = [{', '.join([f'{v:.3f}' for v in pc2])}]")

                    st.markdown("**Mean vector** (center of the vocabulary):")
                    st.code(f"mean = [{', '.join([f'{v:.3f}' for v in mean_vec])}]")

                    st.markdown("---")
                    st.markdown("### Example: Projecting a word to 2D")

                    # Pick first word from the background for example
                    example_idx = 0
                    example_word = space_data['words'][example_idx]
                    example_emb = st.session_state.embeddings[space_data['indices'][example_idx]]
                    example_2d = space_data['reduced'][example_idx]

                    st.markdown(f"**Word:** `{example_word}`")
                    st.code(f"embedding = [{', '.join([f'{v:.3f}' for v in example_emb])}]")

                    # Center the embedding
                    centered = example_emb - mean_vec
                    st.markdown("**Step 1: Center the embedding**")
                    st.code(f"centered = embedding - mean = [{', '.join([f'{v:.3f}' for v in centered])}]")

                    # Project onto PC1
                    proj1 = np.dot(centered, pc1)
                    st.markdown("**Step 2: Project onto PC1**")
                    st.latex(rf"x_1 = \text{{centered}} \cdot \text{{PC1}} = \sum_{{i=1}}^{{{len(centered)}}} c_i \times \text{{PC1}}_i = {proj1:.6f}")

                    # Project onto PC2
                    proj2 = np.dot(centered, pc2)
                    st.markdown("**Step 3: Project onto PC2**")
                    st.latex(rf"x_2 = \text{{centered}} \cdot \text{{PC2}} = \sum_{{i=1}}^{{{len(centered)}}} c_i \times \text{{PC2}}_i = {proj2:.6f}")

                    st.markdown("**Final 2D coordinates:**")
                    st.latex(rf"\begin{{bmatrix}} x_1 \\ x_2 \end{{bmatrix}} = \begin{{bmatrix}} {example_2d[0]:.6f} \\ {example_2d[1]:.6f} \end{{bmatrix}}")

                    st.success(f"✅ This is the position of '{example_word}' in the 2D plot above!")

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback

            st.code(traceback.format_exc())

else:
    st.info("👈 Loading model...")

