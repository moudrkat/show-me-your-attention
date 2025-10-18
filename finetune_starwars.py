"""
Fine-tuning script for TinyStories model on Star Wars stories.

This script fine-tunes a pre-trained TinyStories model to generate
Star Wars-themed children's stories.

Usage:
    python finetune_starwars.py
    python finetune_starwars.py --model roneneldan/TinyStories-8M
    python finetune_starwars.py --use-lora  # Parameter-efficient training
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from torch.utils.data import DataLoader

# Import configuration
from config.training_config import TrainingConfig, StarWarsPrompts

# Optional: LoRA for parameter-efficient fine-tuning
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft library not available. LoRA fine-tuning will not work.")
    print("Install with: pip install peft")


class StarWarsFinetuner:
    """Fine-tune TinyStories model on Star Wars stories."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        set_seed(config.seed)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model and tokenizer
        print(f"\nLoading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.use_fp16 and torch.cuda.is_available() else torch.float32
        )

        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Apply LoRA if requested
        if config.use_lora:
            self.apply_lora()

        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")

    def apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning."""
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for LoRA. Install with: pip install peft")

        print("\nApplying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def load_dataset(self) -> DatasetDict:
        """Load and prepare the Star Wars stories dataset."""
        print(f"\nLoading dataset from: {self.config.dataset_path}")

        dataset_path = Path(self.config.dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}\n"
                f"Please run: python prepare_starwars_dataset.py"
            )

        # Load the JSON file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)

        print(f"Loaded {len(stories)} stories")

        # Create HuggingFace dataset
        dataset = Dataset.from_list(stories)

        # Split into train and validation
        split_dataset = dataset.train_test_split(
            test_size=1.0 - self.config.train_test_split,
            seed=self.config.seed,
            shuffle=True
        )

        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })

        print(f"Train set: {len(dataset_dict['train'])} examples")
        print(f"Validation set: {len(dataset_dict['validation'])} examples")

        return dataset_dict

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize the stories for causal language modeling.

        Args:
            examples: Batch of examples with 'text' field

        Returns:
            Tokenized examples with input_ids, attention_mask, and labels
        """
        # Tokenize the texts
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_datasets(self, dataset_dict: DatasetDict) -> DatasetDict:
        """
        Tokenize and prepare datasets for training.

        Args:
            dataset_dict: Raw text datasets

        Returns:
            Tokenized datasets
        """
        print("\nTokenizing datasets...")

        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            desc="Tokenizing",
        )

        print("Tokenization complete!")

        return tokenized_datasets

    def train(self):
        """Run the fine-tuning process."""
        # Load and prepare datasets
        dataset_dict = self.load_dataset()
        tokenized_datasets = self.prepare_datasets(dataset_dict)

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            logging_dir=str(output_dir / "logs"),
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.use_fp16 and torch.cuda.is_available(),
            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # Change to "wandb" or "tensorboard" if you want logging
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
        )

        # Train!
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        trainer.train()

        # Save the final model
        print("\nSaving final model...")
        trainer.save_model(str(output_dir / "final"))
        self.tokenizer.save_pretrained(str(output_dir / "final"))

        print(f"\n✓ Training complete! Model saved to: {output_dir / 'final'}")

        # Evaluate on test set
        print("\nEvaluating on validation set...")
        eval_results = trainer.evaluate()
        print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
        print(f"Final validation perplexity: {np.exp(eval_results['eval_loss']):.2f}")

        return trainer

    def generate_samples(self, model_path: str = None):
        """
        Generate sample stories to test the fine-tuned model.

        Args:
            model_path: Path to the fine-tuned model (uses config.output_dir/final if None)
        """
        if model_path is None:
            model_path = Path(self.config.output_dir) / "final"

        print(f"\nLoading fine-tuned model from: {model_path}")

        # Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(self.device)
        model.eval()

        print("\nGenerating sample Star Wars stories...")
        print("=" * 60)

        # Generate stories from example prompts
        for i, prompt in enumerate(random.sample(StarWarsPrompts.PROMPTS, min(5, len(StarWarsPrompts.PROMPTS)))):
            print(f"\nPrompt {i+1}: {prompt}")
            print("-" * 60)

            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.eval_max_new_tokens,
                    temperature=self.config.eval_temperature,
                    top_p=self.config.eval_top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode and print
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(generated_text)
            print()

        print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune TinyStories on Star Wars stories")

    parser.add_argument(
        "--model",
        type=str,
        default="roneneldan/TinyStories-33M",
        help="Model name or path (default: roneneldan/TinyStories-33M)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/starwars_stories.json",
        help="Path to Star Wars stories dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/tinystories-starwars",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Skip training and only generate samples from existing model"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config from arguments
    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        max_length=args.max_length,
    )

    print("Star Wars Story Fine-tuning")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA: {config.use_lora}")
    print("=" * 60)

    # Initialize fine-tuner
    finetuner = StarWarsFinetuner(config)

    if args.generate_only:
        # Only generate samples from existing model
        finetuner.generate_samples()
    else:
        # Train the model
        finetuner.train()

        # Generate samples after training
        print("\nGenerating sample stories from fine-tuned model...")
        finetuner.generate_samples()


if __name__ == "__main__":
    main()
