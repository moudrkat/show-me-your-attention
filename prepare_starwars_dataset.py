"""
Dataset preparation script for Star Wars stories.
This script helps you create and format a dataset for fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, DatasetDict


class StarWarsDatasetPreparer:
    """Prepare Star Wars stories dataset for fine-tuning."""

    def __init__(self, output_path: str = "data/starwars_stories.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def create_example_dataset(self) -> List[Dict[str, str]]:
        """
        Create a small example dataset of Star Wars stories.

        In practice, you would:
        1. Collect real Star Wars stories from books, wikis, or fan fiction
        2. Use a larger language model to generate synthetic stories
        3. Scrape and clean data from Star Wars resources
        4. Use existing datasets like wookieepedia summaries

        Returns:
            List of story dictionaries with 'text' field
        """

        example_stories = [
            {
                "text": "Once upon a time, in a galaxy far, far away, a young Jedi named Luke trained with Master Yoda. "
                        "He learned to use the Force and control his emotions. One day, Yoda told him, 'Do or do not, there is no try.' "
                        "Luke practiced lifting rocks with his mind until he became very strong."
            },
            {
                "text": "On the desert planet of Tatooine, a small droid named R2-D2 rolled across the sand dunes. "
                        "He was carrying a secret message from Princess Leia. The message said, 'Help me, Obi-Wan Kenobi, you're my only hope.' "
                        "R2-D2 needed to find the old Jedi master before the Empire caught him."
            },
            {
                "text": "The Millennium Falcon soared through hyperspace with Han Solo at the controls. "
                        "Chewbacca sat beside him, making worried sounds. 'Don't worry, Chewie,' Han said with a grin. "
                        "'This ship made the Kessel Run in less than twelve parsecs. We'll get to Alderaan safely.'"
            },
            {
                "text": "In the dark corridors of the Death Star, a stormtrooper stood guard. "
                        "Suddenly, he heard a strange noise. It was the sound of a lightsaber! "
                        "Before he could react, a flash of green light appeared, and Jedi Master Luke Skywalker rescued his friends."
            },
            {
                "text": "Princess Leia stood before the Rebel Alliance council. 'The Empire has built a superweapon,' she warned. "
                        "'The Death Star can destroy entire planets. We must find its weakness and destroy it before they use it against us.' "
                        "The rebels listened carefully and began planning their attack."
            },
            {
                "text": "Deep in the swamps of Dagobah, Yoda's small hut sat quietly. "
                        "Inside, the ancient Jedi master cooked a simple meal. He sensed a disturbance in the Force. "
                        "'Someone is coming,' he said to himself. 'Strong with the Force, this one is.'"
            },
            {
                "text": "Darth Vader walked onto the bridge of his Star Destroyer. Officers trembled as he passed. "
                        "'Admiral,' Vader said in his deep mechanical voice, 'have you found the rebel base?' "
                        "The admiral nervously replied, 'We are searching the outer rim, my lord.'"
            },
            {
                "text": "The Ewoks of Endor lived peacefully in their tree villages. "
                        "One day, they discovered a golden droid named C-3PO caught in a trap. "
                        "The Ewoks had never seen anything like him before. They decided to take him to their chief."
            },
            {
                "text": "Young Anakin Skywalker raced his podracer through the dangerous canyon. "
                        "The engines roared as he dodged rocks and other racers. 'This is so wizard!' he shouted with excitement. "
                        "He was the only human who could do this because of his strong connection to the Force."
            },
            {
                "text": "Obi-Wan Kenobi sat in meditation at the Jedi Temple. "
                        "He could feel the living Force flowing through everything around him. "
                        "His master, Qui-Gon Jinn, taught him that the Force guides those who listen. "
                        "Obi-Wan cleared his mind and opened himself to its wisdom."
            },
        ]

        return example_stories

    def load_from_text_file(self, file_path: str, story_separator: str = "\n\n") -> List[Dict[str, str]]:
        """
        Load stories from a text file where stories are separated by a delimiter.

        Args:
            file_path: Path to text file containing stories
            story_separator: String that separates different stories

        Returns:
            List of story dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        stories = content.split(story_separator)
        stories = [s.strip() for s in stories if s.strip()]

        return [{"text": story} for story in stories]

    def load_from_jsonl(self, file_path: str, text_field: str = "text") -> List[Dict[str, str]]:
        """
        Load stories from a JSONL file.

        Args:
            file_path: Path to JSONL file
            text_field: Name of the field containing the story text

        Returns:
            List of story dictionaries
        """
        stories = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if text_field in data:
                    stories.append({"text": data[text_field]})

        return stories

    def augment_stories(self, stories: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Simple data augmentation by adding different story starters.

        Args:
            stories: Original stories

        Returns:
            Augmented stories with variations
        """
        starters = [
            "Once upon a time, ",
            "Long ago, ",
            "In a galaxy far away, ",
            "A long time ago, ",
            "",  # No starter
        ]

        augmented = []
        for story in stories:
            text = story["text"]
            # Add original
            augmented.append({"text": text})

            # Add variations (if story doesn't already have a starter)
            if not any(text.startswith(s) for s in starters[:-1]):
                for starter in random.sample(starters[:-2], min(2, len(starters)-2)):
                    augmented.append({"text": starter + text})

        return augmented

    def save_dataset(self, stories: List[Dict[str, str]], shuffle: bool = True):
        """
        Save stories to JSON file.

        Args:
            stories: List of story dictionaries
            shuffle: Whether to shuffle the stories
        """
        if shuffle:
            random.shuffle(stories)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(stories, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(stories)} stories to {self.output_path}")

    def create_huggingface_dataset(
        self,
        stories: List[Dict[str, str]],
        train_split: float = 0.9
    ) -> DatasetDict:
        """
        Create a HuggingFace DatasetDict with train/validation splits.

        Args:
            stories: List of story dictionaries
            train_split: Proportion of data to use for training

        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        dataset = Dataset.from_list(stories)

        # Split into train and validation
        split_dataset = dataset.train_test_split(
            test_size=1.0 - train_split,
            seed=42,
            shuffle=True
        )

        return DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })

    def validate_dataset(self, stories: List[Dict[str, str]]) -> bool:
        """
        Validate that the dataset is properly formatted.

        Args:
            stories: List of story dictionaries

        Returns:
            True if valid, raises exception otherwise
        """
        if not stories:
            raise ValueError("Dataset is empty!")

        for i, story in enumerate(stories):
            if "text" not in story:
                raise ValueError(f"Story {i} missing 'text' field")

            if not isinstance(story["text"], str):
                raise ValueError(f"Story {i} 'text' field is not a string")

            if len(story["text"]) < 10:
                print(f"Warning: Story {i} is very short ({len(story['text'])} chars)")

        print(f"✓ Dataset validation passed: {len(stories)} stories")
        return True


def main():
    """Example usage of the dataset preparer."""

    print("Star Wars Dataset Preparer")
    print("=" * 50)

    preparer = StarWarsDatasetPreparer(output_path="data/starwars_stories.json")

    # Option 1: Create example dataset (for testing)
    print("\n1. Creating example dataset...")
    stories = preparer.create_example_dataset()

    # Option 2: Uncomment to load from your own text file
    # print("\n2. Loading from text file...")
    # stories = preparer.load_from_text_file("path/to/your/starwars_stories.txt")

    # Option 3: Uncomment to load from JSONL
    # print("\n3. Loading from JSONL...")
    # stories = preparer.load_from_jsonl("path/to/your/starwars_stories.jsonl")

    # Validate the dataset
    print("\nValidating dataset...")
    preparer.validate_dataset(stories)

    # Optionally augment the data
    print("\nAugmenting dataset...")
    stories = preparer.augment_stories(stories)
    print(f"After augmentation: {len(stories)} stories")

    # Save to JSON file
    print("\nSaving dataset...")
    preparer.save_dataset(stories)

    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset splits...")
    dataset_dict = preparer.create_huggingface_dataset(stories, train_split=0.9)
    print(f"Train set: {len(dataset_dict['train'])} examples")
    print(f"Validation set: {len(dataset_dict['validation'])} examples")

    # Show a sample
    print("\nSample story:")
    print("-" * 50)
    print(dataset_dict['train'][0]['text'])
    print("-" * 50)

    print("\n✓ Dataset preparation complete!")
    print(f"  Dataset saved to: {preparer.output_path}")
    print(f"  Total stories: {len(stories)}")
    print(f"\nNext step: Run finetune_starwars.py to start training")


if __name__ == "__main__":
    main()
