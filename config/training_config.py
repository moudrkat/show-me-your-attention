"""
Training configuration for fine-tuning TinyStories models on Star Wars stories.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning TinyStories model."""

    # Model configuration
    model_name: str = "roneneldan/TinyStories-33M"  # or "roneneldan/TinyStories-8M"

    # Dataset configuration
    dataset_path: str = "data/starwars_stories.json"  # Path to your Star Wars stories
    max_length: int = 512  # Maximum sequence length
    train_test_split: float = 0.9  # 90% train, 10% validation

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Optimizer settings
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Logging and checkpointing
    output_dir: str = "models/tinystories-starwars"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only last 3 checkpoints

    # Generation settings for evaluation
    eval_max_new_tokens: int = 200
    eval_temperature: float = 0.8
    eval_top_p: float = 0.9

    # Hardware settings
    use_fp16: bool = True  # Mixed precision training
    dataloader_num_workers: int = 2

    # Seed for reproducibility
    seed: int = 42

    # LoRA settings (optional - for parameter-efficient fine-tuning)
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will default to ["q_proj", "v_proj"]

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Default LoRA target modules for GPT-style models
            self.lora_target_modules = ["c_attn", "c_proj"]


@dataclass
class StarWarsPrompts:
    """Example prompts for Star Wars story generation."""

    PROMPTS = [
        "Once upon a time, in a galaxy far, far away, a young Jedi",
        "The Millennium Falcon soared through hyperspace as",
        "On the desert planet of Tatooine,",
        "Master Yoda looked at his young padawan and said,",
        "The Death Star loomed ominously over the planet as",
        "In the Jedi Temple, younglings were learning",
        "A stormtrooper patrolled the corridors of the Star Destroyer when",
        "Princess Leia sent a secret message to",
        "The lightsaber ignited with a brilliant blue glow as",
        "Deep in the forests of Endor, the Ewoks discovered",
    ]
