"""
Show Me Your Attention: Visualizing LLM attention patterns and neuron activations.
"""

from .model_loader import AttentionExtractor
from .visualizer import AttentionVisualizer

__version__ = "0.1.0"
__all__ = ["AttentionExtractor", "AttentionVisualizer"]
