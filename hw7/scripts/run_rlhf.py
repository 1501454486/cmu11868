#!/usr/bin/env python3
"""
Script to run RLHF training using VERL for Assignment 7.
This script loads a trained reward model and runs PPO training on a language model.
"""

import os
import sys
import argparse
import torch
import time
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import AssignmentConfig, get_config, load_config_from_file
from reward_model import load_reward_model
from rlhf_trainer import create_rlhf_trainer, evaluate_policy
from utils import (
    load_json_data,
    setup_logging,
    set_seed,
    MetricsTracker,
    plot_training_curves,
    plot_reward_distribution,
    compare_models_side_by_side,
    create_summary_report,
    save_model_outputs,
    create_sample_data_files
)

logger = logging.getLogger(__name__)


def run_rlhf_training(config: AssignmentConfig):
    """
    Main function to run RLHF training.
    
    Args:
        config: Training configuration
    """
    # Set up logging
    logger = setup_logging(config.system.logs_dir, "rlhf_training")
    
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Set device
    device = torch.device(config.system.device)
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.system.rlhf_model_dir, exist_ok=True)
    os.makedirs(config.system.logs_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Check if data files exist
    if not os.path.exists(config.data.train_prompts_path):
        logger.info("Training data not found, creating sample data...")
        create_sample_data_files("data")
    
    # Load training and evaluation prompts
    logger.info("Loading training data...")
    try:
        train_prompts_data = load_json_data(config.data.train_prompts_path)
        eval_prompts_data = load_json_data(config.data.eval_prompts_path)
    except FileNotFoundError as e:
        logger.error(f"Could not find data files: {e}")
        logger.error("Please run: python src/utils.py to create sample data")
        return
    
    # Extract prompts from data
    train_prompts = [item['prompt'] for item in train_prompts_data]
    eval_prompts = [item['prompt'] for item in eval_prompts_data]
    
    logger.info(f"Loaded {len(train_prompts)} training prompts")
    logger.info(f"Loaded {len(eval_prompts)} evaluation prompts")
    
    # Load reward model
    reward_model_path = os.path.join(config.system.reward_model_dir, "best_reward_model.pt")
    if not os.path.exists(reward_model_path):
        reward_model_path = os.path.join(config.system.reward_model_dir, "final_reward_model.pt")
    
    if not os.path.exists(reward_model_path):
        logger.error(f"Could not find trained reward model at {reward_model_path}")
        logger.error("Please run: python scripts/train_reward_model.py first")
        return
    
    logger.info(f"Loading reward model from {reward_model_path}")
    reward_model = load_reward_model(reward_model_path, device)
    
    # Create RLHF trainer
    logger.info(f"Creating RLHF trainer with model: {config.model.model_name}")
    trainer = create_rlhf_trainer(
        model_name=config.model.model_name,
        reward_model=reward_model,
        config=config,
        device=device
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Evaluate base model before training
    logger.info("Evaluating base model...")
    base_eval_metrics = evaluate_policy(
        trainer=trainer,
        eval_prompts=eval_prompts,
        num_samples=3
    )
    
    logger.info("Base model evaluation results:")
    for key, value in base_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Generate baseline samples for comparison
    logger.info("Generating baseline samples...")
    baseline_responses, _ = trainer.policy.generate(
        prompts=eval_prompts[:5],  # First 5 prompts for comparison
        max_length=config.verl.rollout_max_length,
        temperature=config.verl.rollout_temperature,
        top_p=config.verl.rollout_top_p
    )
    
    baseline_full_texts = [f"{prompt} {response}" 
                          for prompt, response in zip(eval_prompts[:5], baseline_responses)]
    baseline_rewards = reward_model.get_rewards(baseline_full_texts, trainer.tokenizer, device)
    
    # Start RLHF training
    logger.info("Starting RLHF training...")
    start_time = time.time()
    
    best_reward = float('-inf')
    total_steps = 0
    
    # Training loop
    for epoch in range(config.training.ppo_num_epochs):
        logger.info(f"RLHF Epoch {epoch + 1}/{config.training.ppo_num_epochs}")
        
        epoch_metrics = {
            'epoch': epoch,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_divergence': 0.0,
            'reward_mean': 0.0,
            'reward_std': 0.0
        }
        
        num_batches = len(train_prompts) // config.verl.rollout_batch_size
        if len(train_prompts) % config.verl.rollout_batch_size != 0:
            num_batches += 1
        
        # Create progress bar for epoch
        epoch_progress = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
        
        for batch_idx in epoch_progress:
            # Get batch of prompts
            start_idx = batch_idx * config.verl.rollout_batch_size
            end_idx = min(start_idx + config.verl.rollout_batch_size, len(train_prompts))
            batch_prompts = train_prompts[start_idx:end_idx]
            
            # Generate rollouts
            rollout_batch = trainer.generate_rollouts(batch_prompts)
            
            # Training step
            step_metrics = trainer.train_step(rollout_batch)
            
            # Update epoch metrics
            for key in epoch_metrics:
                if key != 'epoch' and hasattr(step_metrics, key):
                    epoch_metrics[key] += getattr(step_metrics, key)
            
            total_steps += 1
            
            # Update progress bar
            epoch_progress.set_postfix({
                'reward': f"{step_metrics.reward_mean:.3f}",
                'policy_loss': f"{step_metrics.policy_loss:.4f}",
                'kl': f"{step_metrics.kl_divergence:.4f}"
            })
            
            # Log detailed metrics periodically
            if total_steps % config.system.logging_steps == 0:
                step_metrics_dict = {
                    'step': total_steps,
                    'policy_loss': step_metrics.policy_loss,
                    'value_loss': step_metrics.value_loss,
                    'entropy': step_metrics.entropy,
                    'kl_divergence': step_metrics.kl_divergence,
                    'reward_mean': step_metrics.reward_mean,
                    'reward_std': step_metrics.reward_std,
                    'advantage_mean': step_metrics.advantage_mean,
                    'advantage_std': step_metrics.advantage_std
                }
                
                metrics_tracker.update(step_metrics_dict, step=total_steps)
                
                logger.info(f"Step {total_steps}:")
                logger.info(f"  Policy Loss: {step_metrics.policy_loss:.4f}")
                logger.info(f"  Value Loss: {step_metrics.value_loss:.4f}")
                logger.info(f"  Reward Mean: {step_metrics.reward_mean:.4f}")
                logger.info(f"  KL Divergence: {step_metrics.kl_divergence:.4f}")
        
        # Average epoch metrics
        for key in epoch_metrics:
            if key != 'epoch':
                epoch_metrics[key] /= num_batches
        
        # Evaluate model after epoch
        if (epoch + 1) % config.system.eval_steps == 0 or epoch == config.training.ppo_num_epochs - 1:
            logger.info(f"Evaluating model after epoch {epoch + 1}...")
            eval_metrics = evaluate_policy(
                trainer=trainer,
                eval_prompts=eval_prompts,
                num_samples=3
            )
            
            # Add evaluation metrics to epoch metrics
            for key, value in eval_metrics.items():
                epoch_metrics[f'eval_{key}'] = value
            
            logger.info("Evaluation results:")
            for key, value in eval_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Check if this is the best model so far
            current_reward = eval_metrics['mean_reward']
            if current_reward > best_reward:
                best_reward = current_reward
                
                # Save best model
                best_model_path = os.path.join(
                    config.system.rlhf_model_dir,
                    "best_rlhf_model"
                )
                trainer.policy.model.save_pretrained(best_model_path)
                trainer.tokenizer.save_pretrained(best_model_path)
                
                logger.info(f"Saved new best model with reward: {current_reward:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.system.save_steps == 0:
            checkpoint_path = os.path.join(
                config.system.rlhf_model_dir,
                f"rlhf_checkpoint_epoch_{epoch + 1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
        
        # Update metrics tracker with epoch metrics
        metrics_tracker.update(epoch_metrics, step=epoch)
        
        logger.info(f"Epoch {epoch + 1} completed:")
        logger.info(f"  Average Reward: {epoch_metrics['reward_mean']:.4f}")
        logger.info(f"  Policy Loss: {epoch_metrics['policy_loss']:.4f}")
        logger.info(f"  KL Divergence: {epoch_metrics['kl_divergence']:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"RLHF training completed in {training_time:.2f} seconds")
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_eval_metrics = evaluate_policy(
        trainer=trainer,
        eval_prompts=eval_prompts,
        num_samples=5
    )
    
    logger.info("Final evaluation results:")
    for key, value in final_eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Generate final samples for comparison
    logger.info("Generating final samples for comparison...")
    final_responses, _ = trainer.policy.generate(
        prompts=eval_prompts[:5],
        max_length=config.verl.rollout_max_length,
        temperature=config.verl.rollout_temperature,
        top_p=config.verl.rollout_top_p
    )
    
    final_full_texts = [f"{prompt} {response}" 
                       for prompt, response in zip(eval_prompts[:5], final_responses)]
    final_rewards = reward_model.get_rewards(final_full_texts, trainer.tokenizer, device)
    
    # Save models and results
    logger.info("Saving final models and results...")
    
    # Save final model
    final_model_path = os.path.join(config.system.rlhf_model_dir, "final_rlhf_model")
    trainer.policy.model.save_pretrained(final_model_path)
    trainer.tokenizer.save_pretrained(final_model_path)
    
    # Save metrics
    metrics_tracker.save_metrics(
        os.path.join(config.system.logs_dir, "rlhf_training_metrics.json")
    )
    
    # Create plots
    plot_training_curves(
        metrics_tracker.metrics_history,
        os.path.join("plots", "rlhf_training_curves.png"),
        title="RLHF Training Progress"
    )
    
    # Plot reward distributions
    all_rewards = []
    for metrics in metrics_tracker.metrics_history:
        if 'reward_mean' in metrics:
            all_rewards.append(metrics['reward_mean'])
    
    if all_rewards:
        plot_reward_distribution(
            all_rewards,
            os.path.join("plots", "reward_distribution.png"),
            title="Reward Distribution During Training"
        )
    
    # Save model outputs for analysis
    save_model_outputs(
        prompts=eval_prompts[:5],
        responses=baseline_responses,
        rewards=baseline_rewards,
        save_path=os.path.join(config.system.logs_dir, "baseline_outputs.json"),
        model_name="baseline"
    )
    
    save_model_outputs(
        prompts=eval_prompts[:5],
        responses=final_responses,
        rewards=final_rewards,
        save_path=os.path.join(config.system.logs_dir, "rlhf_outputs.json"),
        model_name="rlhf"
    )
    
    # Create comparison
    compare_models_side_by_side(
        prompts=eval_prompts[:5],
        base_responses=baseline_responses,
        rlhf_responses=final_responses,
        base_rewards=baseline_rewards,
        rlhf_rewards=final_rewards,
        save_path=os.path.join(config.system.logs_dir, "model_comparison.json")
    )
    
    # Create summary report
    create_summary_report(
        base_metrics=base_eval_metrics,
        rlhf_metrics=final_eval_metrics,
        training_time=training_time,
        save_path=os.path.join(config.system.logs_dir, "training_summary.json")
    )
    
    logger.info("RLHF training completed successfully!")
    logger.info(f"Base model mean reward: {base_eval_metrics['mean_reward']:.4f}")
    logger.info(f"RLHF model mean reward: {final_eval_metrics['mean_reward']:.4f}")
    logger.info(f"Improvement: {final_eval_metrics['mean_reward'] - base_eval_metrics['mean_reward']:.4f}")
    logger.info(f"Models saved to: {config.system.rlhf_model_dir}")
    logger.info(f"Logs and plots saved to: {config.system.logs_dir} and plots/")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run RLHF training with VERL")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Name of the base language model"
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=None,
        help="Path to trained reward model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for PPO training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of PPO training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Rollout batch size"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    if args.model_name:
        config.model.model_name = args.model_name
    if args.learning_rate:
        config.training.ppo_learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.ppo_num_epochs = args.num_epochs
    if args.batch_size:
        config.verl.rollout_batch_size = args.batch_size
    if args.reward_model_path:
        config.system.reward_model_dir = os.path.dirname(args.reward_model_path)
    
    # Validate configuration
    config.validate()
    
    # Run RLHF training
    run_rlhf_training(config)


if __name__ == "__main__":
    main()