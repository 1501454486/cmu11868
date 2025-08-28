#!/usr/bin/env python3
"""
Data Preparation Script for Assignment 7.
This script downloads and prepares the Anthropic HH-RLHF dataset from Hugging Face.
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Datasets library not available. Please install with: pip install datasets")

from utils import save_json_data


def load_hh_rlhf_dataset(max_samples: int = 1000) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and process the Anthropic HH-RLHF dataset.
    
    Args:
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    print("📥 Loading Anthropic HH-RLHF dataset from Hugging Face...")
    
    # Load the dataset
    try:
        # Load both helpful and harmless subsets
        helpful_dataset = load_dataset("Anthropic/hh-rlhf", "helpful-base", split="train")
        harmless_dataset = load_dataset("Anthropic/hh-rlhf", "harmless-base", split="train")
        
        print(f"Loaded {len(helpful_dataset)} helpful examples")
        print(f"Loaded {len(harmless_dataset)} harmless examples")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Falling back to sample data...")
        return create_fallback_data(max_samples)
    
    # Combine and shuffle datasets
    all_data = []
    
    # Process helpful examples
    for i, example in enumerate(helpful_dataset):
        if len(all_data) >= max_samples:
            break
            
        processed_example = process_hh_example(example)
        if processed_example:
            all_data.append(processed_example)
    
    # Add harmless examples if we need more
    remaining_samples = max_samples - len(all_data)
    if remaining_samples > 0:
        for i, example in enumerate(harmless_dataset):
            if len(all_data) >= max_samples:
                break
                
            processed_example = process_hh_example(example)
            if processed_example:
                all_data.append(processed_example)
    
    # Shuffle the combined data
    random.shuffle(all_data)
    
    print(f"✅ Processed {len(all_data)} total examples")
    
    # Split into train/validation
    split_idx = int(len(all_data) * 0.9)  # 90% train, 10% validation
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    return train_data, val_data


def process_hh_example(example: Dict) -> Dict:
    """
    Process a single example from the HH-RLHF dataset.
    
    Args:
        example: Raw example from the dataset
        
    Returns:
        Processed example in our format
    """
    try:
        # Extract the conversation components
        chosen = example.get('chosen', '')
        rejected = example.get('rejected', '')
        
        if not chosen or not rejected:
            return None
        
        # Extract the prompt (everything before the last Assistant response)
        # The format is typically: "Human: ... Assistant: ... Human: ... Assistant:"
        # We want to extract up to the last "Assistant:" as the prompt
        
        # Find the last occurrence of "Assistant:" in chosen response
        last_assistant_idx = chosen.rfind("Assistant:")
        if last_assistant_idx == -1:
            return None
        
        # Extract prompt (everything up to and including the last "Assistant:")
        prompt = chosen[:last_assistant_idx + len("Assistant:")].strip()
        
        # Extract the actual responses (everything after "Assistant:")
        chosen_response = chosen[last_assistant_idx + len("Assistant:"):].strip()
        rejected_response = rejected[last_assistant_idx + len("Assistant:"):].strip()
        
        # Skip if responses are too short or too long
        if len(chosen_response) < 10 or len(chosen_response) > 500:
            return None
        if len(rejected_response) < 10 or len(rejected_response) > 500:
            return None
        
        return {
            'prompt': prompt,
            'chosen': prompt + " " + chosen_response,
            'rejected': prompt + " " + rejected_response
        }
        
    except Exception as e:
        print(f"⚠️  Error processing example: {e}")
        return None


def create_fallback_data(max_samples: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Create fallback data if the main dataset cannot be loaded.
    
    Args:
        max_samples: Number of samples to create
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    print("🔄 Creating fallback sample data...")
    
    # Sample conversations in HH-RLHF format
    sample_data = [
        {
            'prompt': 'Human: How can I be more productive at work?\n\nAssistant:',
            'chosen': 'Human: How can I be more productive at work?\n\nAssistant: Here are some evidence-based strategies for improving workplace productivity:\n\n1. **Prioritize tasks** using methods like the Eisenhower Matrix to focus on important and urgent work\n2. **Time-block your calendar** to dedicate specific periods to focused work\n3. **Minimize distractions** by turning off non-essential notifications\n4. **Take regular breaks** to maintain focus and prevent burnout\n5. **Use productivity tools** like task managers and note-taking apps\n\nThe key is finding what works for your specific role and work style, then being consistent with those practices.',
            'rejected': 'Human: How can I be more productive at work?\n\nAssistant: Just work harder and longer hours. Productivity is about grinding through tasks and never taking breaks. Sleep is for the weak.'
        },
        {
            'prompt': 'Human: What\'s the best way to learn a new programming language?\n\nAssistant:',
            'chosen': 'Human: What\'s the best way to learn a new programming language?\n\nAssistant: Here\'s an effective approach to learning a new programming language:\n\n1. **Start with fundamentals** - Learn the basic syntax, data types, and control structures\n2. **Practice regularly** - Code a little bit every day rather than long cramming sessions\n3. **Build projects** - Start with simple programs and gradually increase complexity\n4. **Read others\' code** - Study well-written code to learn best practices\n5. **Join communities** - Participate in forums, Discord servers, or local meetups\n6. **Use official documentation** - Get comfortable reading and using the language\'s docs\n\nThe key is balancing theory with hands-on practice. Don\'t just read about programming - write code!',
            'rejected': 'Human: What\'s the best way to learn a new programming language?\n\nAssistant: Just memorize all the syntax and functions. Once you know all the commands, you\'ll be a great programmer. Don\'t worry about actually writing code or understanding concepts.'
        },
        {
            'prompt': 'Human: I\'m feeling overwhelmed with my workload. Any advice?\n\nAssistant:',
            'chosen': 'Human: I\'m feeling overwhelmed with my workload. Any advice?\n\nAssistant: I understand that feeling overwhelmed can be really stressful. Here are some strategies that might help:\n\n1. **List everything out** - Write down all your tasks to get them out of your head\n2. **Prioritize ruthlessly** - Identify what\'s truly urgent vs. what can wait\n3. **Break large tasks down** - Divide big projects into smaller, manageable steps\n4. **Delegate when possible** - See if others can help with some responsibilities\n5. **Set boundaries** - It\'s okay to say no to additional requests when you\'re at capacity\n6. **Take care of yourself** - Ensure you\'re getting enough sleep, food, and exercise\n\nRemember, this feeling is temporary. Focus on one task at a time and celebrate small wins along the way.',
            'rejected': 'Human: I\'m feeling overwhelmed with my workload. Any advice?\n\nAssistant: Stop complaining and just work harder. Everyone feels overwhelmed sometimes, you just need to push through it. Don\'t be weak.'
        }
    ]
    
    # Duplicate and vary the sample data to reach max_samples
    extended_data = []
    while len(extended_data) < max_samples:
        for base_example in sample_data:
            if len(extended_data) >= max_samples:
                break
            extended_data.append(base_example.copy())
    
    # Shuffle and split
    random.shuffle(extended_data)
    split_idx = int(len(extended_data) * 0.9)
    
    return extended_data[:split_idx], extended_data[split_idx:]


def create_prompts_from_preferences(preference_data: List[Dict]) -> List[Dict]:
    """
    Extract unique prompts from preference data.
    
    Args:
        preference_data: List of preference examples
        
    Returns:
        List of unique prompts
    """
    unique_prompts = set()
    
    for example in preference_data:
        prompt = example['prompt']
        unique_prompts.add(prompt)
    
    return [{'prompt': prompt} for prompt in sorted(unique_prompts)]


def prepare_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    output_dir: str = "data",
    max_samples: int = 1000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Prepare the dataset for training.
    
    Args:
        dataset_name: Name of the dataset to load
        output_dir: Directory to save processed data
        max_samples: Maximum number of samples to process
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    if dataset_name == "Anthropic/hh-rlhf":
        train_data, val_data = load_hh_rlhf_dataset(max_samples)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet")
    
    # Further split validation data into val/test
    val_split_idx = int(len(val_data) * (val_ratio / (val_ratio + test_ratio)))
    test_data = val_data[val_split_idx:]
    val_data = val_data[:val_split_idx]
    
    print(f"📊 Data splits:")
    print(f"   Training: {len(train_data)} examples")
    print(f"   Validation: {len(val_data)} examples") 
    print(f"   Test: {len(test_data)} examples")
    
    # Create training prompts (unique prompts from training set)
    train_prompts = create_prompts_from_preferences(train_data)
    print(f"   Unique training prompts: {len(train_prompts)}")
    
    # Create evaluation prompts (unique prompts from test set)
    eval_prompts = create_prompts_from_preferences(test_data)
    print(f"   Unique evaluation prompts: {len(eval_prompts)}")
    
    # Save all data files
    save_json_data(train_prompts, os.path.join(output_dir, "train_prompts.json"))
    save_json_data(eval_prompts, os.path.join(output_dir, "eval_prompts.json"))
    save_json_data(train_data, os.path.join(output_dir, "preference_data.json"))
    save_json_data(val_data, os.path.join(output_dir, "preference_data_val.json"))
    save_json_data(test_data, os.path.join(output_dir, "preference_data_test.json"))
    
    # Create a small sample for quick testing
    sample_size = min(10, len(train_data))
    sample_data = train_data[:sample_size]
    save_json_data(sample_data, os.path.join(output_dir, "preference_data_sample.json"))
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Files saved to {output_dir}/:")
    print(f"   - train_prompts.json ({len(train_prompts)} prompts)")
    print(f"   - eval_prompts.json ({len(eval_prompts)} prompts)")
    print(f"   - preference_data.json ({len(train_data)} pairs)")
    print(f"   - preference_data_val.json ({len(val_data)} pairs)")
    print(f"   - preference_data_test.json ({len(test_data)} pairs)")
    print(f"   - preference_data_sample.json ({sample_size} pairs)")


def validate_data(data_dir: str = "data"):
    """
    Validate the prepared data files.
    
    Args:
        data_dir: Directory containing data files
    """
    print("🔍 Validating prepared data files...")
    
    required_files = [
        ("train_prompts.json", ["prompt"]),
        ("eval_prompts.json", ["prompt"]),
        ("preference_data.json", ["prompt", "chosen", "rejected"]),
    ]
    
    all_valid = True
    
    for filename, required_keys in required_files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            all_valid = False
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"{filename}: Data should be a list")
                all_valid = False
                continue
            
            if len(data) == 0:
                print(f"{filename}: File is empty")
                all_valid = False
                continue
            
            # Check required keys
            for i, item in enumerate(data[:5]):  # Check first 5 items
                if not isinstance(item, dict):
                    print(f"{filename}: Item {i} should be a dictionary")
                    all_valid = False
                    break
                
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    print(f"{filename}: Item {i} missing keys: {missing_keys}")
                    all_valid = False
                    break
            else:
                print(f"{filename}: {len(data)} valid items")
                
        except json.JSONDecodeError as e:
            print(f"{filename}: Invalid JSON - {e}")
            all_valid = False
        except Exception as e:
            print(f"{filename}: Error - {e}")
            all_valid = False
    
    if all_valid:
        print("All data files are valid!")
    else:
        print("Some data files have issues. Please run preparation again.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare training data for Assignment 7")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Anthropic/hh-rlhf",
        help="Dataset name from Hugging Face"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for prepared data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing data files instead of preparing new ones"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    if args.validate:
        validate_data(args.output_dir)
    else:
        try:
            prepare_dataset(
                dataset_name=args.dataset,
                output_dir=args.output_dir,
                max_samples=args.max_samples
            )
            print("\nValidating prepared data...")
            validate_data(args.output_dir)
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()