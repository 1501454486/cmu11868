#!/usr/bin/env python3
"""
Script to train the reward model for Assignment 7.
This script loads preference data and trains a reward model using pairwise ranking loss.
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import AssignmentConfig, get_config, load_config_from_file
from reward_model import create_reward_model, RewardModelTrainer, save_reward_model
from utils import (
    load_json_data, 
    split_data, 
    setup_logging, 
    set_seed, 
    MetricsTracker,
    plot_training_curves,
    create_sample_data_files
)

logger = logging.getLogger(__name__)


class PreferenceDataset:
    """
    Dataset class for preference data.
    """
    
    def __init__(self, preference_data):
        """
        Initialize preference dataset.
        
        Args:
            preference_data: List of preference pairs
        """
        self.data = preference_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_data_loader(dataset, batch_size, shuffle=True):
    """
    Create a simple data loader.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        Data loader
    """
    def collate_fn(batch):
        chosen_texts = [item['chosen'] for item in batch]
        rejected_texts = [item['rejected'] for item in batch]
        return {
            'chosen': chosen_texts,
            'rejected': rejected_texts
        }
    
    # Simple batch iterator
    def batch_iterator():
        indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [dataset[idx] for idx in batch_indices]
            yield collate_fn(batch)
    
    return batch_iterator


def train_reward_model(config: AssignmentConfig):
    """
    Main function to train the reward model.
    
    Args:
        config: Training configuration
    """
    logger = setup_logging(config.system.logs_dir, "reward_model_training")
    set_seed(config.experiment.seed)
    device = torch.device(config.system.device)
    logger.info(f"Using device: {device}")
    
    os.makedirs(config.system.reward_model_dir, exist_ok=True)
    os.makedirs(config.system.logs_dir, exist_ok=True)
    
    if not os.path.exists(config.data.preference_data_path):
        logger.info("Preference data not found, creating sample data...")
        create_sample_data_files("data")
    
    logger.info("Loading preference data...")
    try:
        preference_data = load_json_data(config.data.preference_data_path)
    except FileNotFoundError:
        logger.error(f"Could not find preference data at {config.data.preference_data_path}")
        logger.error("Please run: python src/utils.py to create sample data")
        return
    
    required_keys = ['chosen', 'rejected']
    for i, item in enumerate(preference_data):
        for key in required_keys:
            if key not in item:
                logger.error(f"Missing key '{key}' in preference data item {i}")
                return
    
    logger.info(f"Loaded {len(preference_data)} preference pairs")

    train_data, val_data, test_data = split_data(
        preference_data,
        config.data.train_split_ratio,
        config.data.validation_split_ratio,
        config.data.test_split_ratio
    )
    
    train_dataset = PreferenceDataset(train_data)
    val_dataset = PreferenceDataset(val_data)
    test_dataset = PreferenceDataset(test_data)
    
    train_loader = create_data_loader(
        train_dataset, 
        config.data.train_batch_size, 
        shuffle=True
    )
    val_loader = create_data_loader(
        val_dataset, 
        config.data.eval_batch_size, 
        shuffle=False
    )
    
    logger.info(f"Loading tokenizer: {config.model.reward_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.reward_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Creating reward model: {config.model.reward_model_name}")
    model = create_reward_model(
        model_name=config.model.reward_model_name,
        hidden_size=config.model.reward_model_hidden_size,
        dropout=config.model.reward_model_dropout
    )
    
    trainer = RewardModelTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=config.training.reward_learning_rate,
        weight_decay=config.training.reward_weight_decay,
        max_length=config.data.max_prompt_length + config.data.max_response_length
    )
    
    metrics_tracker = MetricsTracker()
    
    logger.info("Starting reward model training...")
    start_time = time.time()
    
    best_val_accuracy = 0.0
    steps_since_improvement = 0
    global_step = 0
    
    for epoch in range(config.training.reward_num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.training.reward_num_epochs}")
        
        model.train()
        train_metrics = {
            'epoch': epoch,
            'train_loss': 0.0,
            'train_accuracy': 0.0,
            'train_chosen_reward_mean': 0.0,
            'train_rejected_reward_mean': 0.0,
            'train_reward_diff': 0.0
        }
        
        train_steps = 0
        train_loader_fn = train_loader()
        
        train_progress = tqdm(
            range(len(train_data) // config.data.train_batch_size + 1),
            desc=f"Training Epoch {epoch + 1}"
        )
        
        for batch in train_progress:
            try:
                batch_data = next(train_loader_fn)
            except StopIteration:
                break
            
            step_metrics = trainer.train_step(batch_data)
            
            for key, value in step_metrics.items():
                train_metrics[f'train_{key}'] += value
            
            train_steps += 1
            global_step += 1
            
            train_progress.set_postfix({
                'loss': step_metrics['loss'],
                'acc': step_metrics['accuracy']
            })
            
            if global_step % config.system.logging_steps == 0:
                current_metrics = {k: v / train_steps for k, v in train_metrics.items() if k != 'epoch'}
                current_metrics['step'] = global_step
                logger.info(f"Step {global_step}: {current_metrics}")
        
        for key in train_metrics:
            if key != 'epoch':
                train_metrics[key] /= train_steps
        
        model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_accuracy': 0.0,
            'val_chosen_reward_mean': 0.0,
            'val_rejected_reward_mean': 0.0,
            'val_reward_diff': 0.0
        }
        
        val_steps = 0
        val_loader_fn = val_loader()
        
        val_progress = tqdm(
            range(len(val_data) // config.data.eval_batch_size + 1),
            desc=f"Validation Epoch {epoch + 1}"
        )
        
        with torch.no_grad():
            for batch in val_progress:
                try:
                    batch_data = next(val_loader_fn)
                except StopIteration:
                    break
                
                step_metrics = trainer.evaluate_step(batch_data)
                
                for key, value in step_metrics.items():
                    val_metrics[f'val_{key}'] += value
                
                val_steps += 1
                
                val_progress.set_postfix({
                    'loss': step_metrics['loss'],
                    'acc': step_metrics['accuracy']
                })
        
        for key in val_metrics:
            val_metrics[key] /= val_steps
        
        epoch_metrics = {**train_metrics, **val_metrics}
        metrics_tracker.update(epoch_metrics, step=epoch)
        
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        
        current_val_accuracy = val_metrics['val_accuracy']
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            steps_since_improvement = 0
            
            best_model_path = os.path.join(
                config.system.reward_model_dir, 
                "best_reward_model.pt"
            )
            save_reward_model(
                model=model,
                model_path=best_model_path,
                tokenizer=tokenizer,
                additional_info={
                    'epoch': epoch,
                    'val_accuracy': current_val_accuracy,
                    'config': config.to_dict()
                }
            )
            logger.info(f"Saved new best model with accuracy: {current_val_accuracy:.4f}")
        else:
            steps_since_improvement += 1
        
        if (epoch + 1) % config.system.save_steps == 0:
            checkpoint_path = os.path.join(
                config.system.reward_model_dir,
                f"reward_model_epoch_{epoch + 1}.pt"
            )
            save_reward_model(
                model=model,
                model_path=checkpoint_path,
                tokenizer=tokenizer,
                additional_info={
                    'epoch': epoch,
                    'val_accuracy': current_val_accuracy,
                    'config': config.to_dict()
                }
            )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    logger.info("Evaluating on test set...")
    test_loader_fn = create_data_loader(test_dataset, config.data.eval_batch_size, shuffle=False)()
    
    test_metrics = {
        'test_loss': 0.0,
        'test_accuracy': 0.0,
        'test_chosen_reward_mean': 0.0,
        'test_rejected_reward_mean': 0.0,
        'test_reward_diff': 0.0
    }
    
    test_steps = 0
    model.eval()
    
    with torch.no_grad():
        for batch_data in test_loader_fn:
            try:
                step_metrics = trainer.evaluate_step(batch_data)
                
                for key, value in step_metrics.items():
                    test_metrics[f'test_{key}'] += value
                
                test_steps += 1
            except StopIteration:
                break
    
    for key in test_metrics:
        test_metrics[key] /= test_steps
    
    logger.info("Test Results:")
    logger.info(f"  Test Loss: {test_metrics['test_loss']:.4f}")
    logger.info(f"  Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    
    metrics_tracker.save_metrics(
        os.path.join(config.system.logs_dir, "reward_model_metrics.json")
    )
    
    plot_training_curves(
        metrics_tracker.metrics_history,
        os.path.join(config.system.logs_dir, "reward_model_training_curves.png"),
        title="Reward Model Training"
    )
    
    final_model_path = os.path.join(
        config.system.reward_model_dir,
        "final_reward_model.pt"
    )
    save_reward_model(
        model=model,
        model_path=final_model_path,
        tokenizer=tokenizer,
        additional_info={
            'training_time': training_time,
            'best_val_accuracy': best_val_accuracy,
            'test_metrics': test_metrics,
            'config': config.to_dict()
        }
    )
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Final test accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"Models saved to: {config.system.reward_model_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train reward model for RLHF")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Name of the base model for reward model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.model_name:
        config.model.reward_model_name = args.model_name
    if args.learning_rate:
        config.training.reward_learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.reward_num_epochs = args.num_epochs
    if args.batch_size:
        config.data.train_batch_size = args.batch_size
    
    config.validate()
    
    train_reward_model(config)


if __name__ == "__main__":
    main()