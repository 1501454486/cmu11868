#!/usr/bin/env python3
"""
Evaluation script for Assignment 7.
This script evaluates and compares the base model and RLHF-trained model.
"""

import os
import sys
import argparse
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import AssignmentConfig, get_config, load_config_from_file
from reward_model import load_reward_model
from utils import (
    load_json_data,
    setup_logging,
    set_seed,
    compute_text_statistics,
    plot_reward_distribution,
    compare_models_side_by_side,
    save_model_outputs,
    create_summary_report,
    create_sample_data_files
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Class for evaluating language models with reward model scoring.
    """
    
    def __init__(self, reward_model, tokenizer, device):
        """
        Initialize the evaluator.
        
        Args:
            reward_model: Trained reward model for scoring
            tokenizer: Tokenizer for text processing
            device: Device for computation
        """
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.reward_model.eval()
    
    def evaluate_model(
        self,
        model,
        model_tokenizer,
        prompts,
        num_samples_per_prompt=5,
        max_length=256,
        temperature=0.7,
        top_p=0.9
    ):
        """
        Evaluate a model on a set of prompts.
        
        Args:
            model: Language model to evaluate
            model_tokenizer: Tokenizer for the model
            prompts: List of prompts to evaluate on
            num_samples_per_prompt: Number of samples to generate per prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        
        all_prompts = []
        all_responses = []
        all_rewards = []
        all_response_lengths = []
        
        logger.info(f"Evaluating model on {len(prompts)} prompts...")
        
        for prompt in tqdm(prompts, desc="Evaluating"):
            # Generate multiple samples for this prompt
            prompt_samples = [prompt] * num_samples_per_prompt
            
            # Tokenize prompts
            encoded = model_tokenizer(
                prompt_samples,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length // 2
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate responses
            with torch.no_grad():
                generated = model.generate(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=model_tokenizer.pad_token_id,
                    eos_token_id=model_tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Extract responses (remove prompt part)
            prompt_length = encoded['input_ids'].shape[1]
            response_ids = generated[:, prompt_length:]
            
            responses = model_tokenizer.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Get rewards for generated responses
            full_texts = [f"{prompt} {response}" for response in responses]
            rewards = self.reward_model.get_rewards(full_texts, self.tokenizer, self.device)
            
            # Compute response statistics
            response_lengths = [len(response.split()) for response in responses]
            
            # Store results
            all_prompts.extend([prompt] * len(responses))
            all_responses.extend(responses)
            all_rewards.extend(rewards)
            all_response_lengths.extend(response_lengths)
        
        # Compute aggregate metrics
        results = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'median_reward': np.median(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'mean_response_length': np.mean(all_response_lengths),
            'std_response_length': np.std(all_response_lengths),
            'num_prompts': len(prompts),
            'num_samples': len(all_responses),
            'samples_per_prompt': num_samples_per_prompt
        }
        
        # Add percentile information
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            results[f'reward_p{p}'] = np.percentile(all_rewards, p)
        
        # Text statistics
        text_stats = compute_text_statistics(all_responses, self.tokenizer)
        results.update({f'response_{k}': v for k, v in text_stats.items()})
        
        return results, all_prompts, all_responses, all_rewards
    
    def compare_models(
        self,
        base_model,
        base_tokenizer,
        rlhf_model,
        rlhf_tokenizer,
        prompts,
        num_samples_per_prompt=3,
        **generation_kwargs
    ):
        """
        Compare base model and RLHF model side by side.
        
        Args:
            base_model: Base language model
            base_tokenizer: Tokenizer for base model
            rlhf_model: RLHF-trained model
            rlhf_tokenizer: Tokenizer for RLHF model
            prompts: List of prompts for comparison
            num_samples_per_prompt: Number of samples per prompt
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Comparison results
        """
        logger.info("Evaluating base model...")
        base_results, base_prompts, base_responses, base_rewards = self.evaluate_model(
            base_model, base_tokenizer, prompts, num_samples_per_prompt, **generation_kwargs
        )
        
        logger.info("Evaluating RLHF model...")
        rlhf_results, rlhf_prompts, rlhf_responses, rlhf_rewards = self.evaluate_model(
            rlhf_model, rlhf_tokenizer, prompts, num_samples_per_prompt, **generation_kwargs
        )
        
        # Compute improvements
        improvements = {}
        for key in base_results:
            if isinstance(base_results[key], (int, float)) and key in rlhf_results:
                base_val = base_results[key]
                rlhf_val = rlhf_results[key]
                
                if base_val != 0:
                    improvement_pct = ((rlhf_val - base_val) / abs(base_val)) * 100
                    improvements[f'{key}_improvement_pct'] = improvement_pct
                
                improvements[f'{key}_improvement_abs'] = rlhf_val - base_val
        
        return {
            'base_results': base_results,
            'rlhf_results': rlhf_results,
            'improvements': improvements,
            'base_data': (base_prompts, base_responses, base_rewards),
            'rlhf_data': (rlhf_prompts, rlhf_responses, rlhf_rewards)
        }


def evaluate_models(config: AssignmentConfig):
    """
    Main evaluation function.
    
    Args:
        config: Evaluation configuration
    """
    # Set up logging
    logger = setup_logging(config.system.logs_dir, "model_evaluation")
    
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Set device
    device = torch.device(config.system.device)
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Check if data files exist
    if not os.path.exists(config.data.eval_prompts_path):
        logger.info("Evaluation data not found, creating sample data...")
        create_sample_data_files("data")
    
    # Load evaluation prompts
    logger.info("Loading evaluation prompts...")
    try:
        eval_prompts_data = load_json_data(config.data.eval_prompts_path)
        eval_prompts = [item['prompt'] for item in eval_prompts_data]
    except FileNotFoundError:
        logger.error(f"Could not find evaluation prompts at {config.data.eval_prompts_path}")
        return
    
    # Limit number of prompts for evaluation if specified
    if hasattr(config.experiment, 'eval_prompts_sample_size') and config.experiment.eval_prompts_sample_size > 0:
        eval_prompts = eval_prompts[:config.experiment.eval_prompts_sample_size]
    
    logger.info(f"Evaluating on {len(eval_prompts)} prompts")
    
    # Load reward model
    reward_model_path = os.path.join(config.system.reward_model_dir, "best_reward_model.pt")
    if not os.path.exists(reward_model_path):
        reward_model_path = os.path.join(config.system.reward_model_dir, "final_reward_model.pt")
    
    if not os.path.exists(reward_model_path):
        logger.error(f"Could not find reward model at {reward_model_path}")
        logger.error("Please train reward model first: python scripts/train_reward_model.py")
        return
    
    logger.info(f"Loading reward model from {reward_model_path}")
    reward_model = load_reward_model(reward_model_path, device)
    
    # Load reward model tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(config.model.reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    # Create evaluator
    evaluator = ModelEvaluator(reward_model, reward_tokenizer, device)
    
    # Load base model
    logger.info(f"Loading base model: {config.model.model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(config.model.model_name)
    
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    
    base_model.to(device)
    
    # Check if RLHF model exists
    rlhf_model_path = os.path.join(config.system.rlhf_model_dir, "best_rlhf_model")
    if not os.path.exists(rlhf_model_path):
        rlhf_model_path = os.path.join(config.system.rlhf_model_dir, "final_rlhf_model")
    
    if os.path.exists(rlhf_model_path):
        # Load RLHF model for comparison
        logger.info(f"Loading RLHF model from {rlhf_model_path}")
        rlhf_tokenizer = AutoTokenizer.from_pretrained(rlhf_model_path)
        rlhf_model = AutoModelForCausalLM.from_pretrained(rlhf_model_path)
        
        if rlhf_tokenizer.pad_token is None:
            rlhf_tokenizer.pad_token = rlhf_tokenizer.eos_token
            rlhf_model.config.pad_token_id = rlhf_tokenizer.pad_token_id
        
        rlhf_model.to(device)
        
        # Compare models
        logger.info("Comparing base model vs RLHF model...")
        comparison_results = evaluator.compare_models(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            rlhf_model=rlhf_model,
            rlhf_tokenizer=rlhf_tokenizer,
            prompts=eval_prompts,
            num_samples_per_prompt=3,
            max_length=config.experiment.eval_generation_max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        # Log comparison results
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        
        logger.info("Base Model Results:")
        for key, value in comparison_results['base_results'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\nRLHF Model Results:")
        for key, value in comparison_results['rlhf_results'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\nImprovements:")
        for key, value in comparison_results['improvements'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
        
        # Save detailed results
        import json
        with open("evaluation_results/detailed_comparison.json", 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for k, v in comparison_results.items():
                if k in ['base_data', 'rlhf_data']:
                    continue  # Skip the large data tuples
                if isinstance(v, dict):
                    serializable_results[k] = {
                        key: float(val) if isinstance(val, (np.float32, np.float64)) else val
                        for key, val in v.items()
                    }
                else:
                    serializable_results[k] = v
            
            json.dump(serializable_results, f, indent=2)
        
        # Extract data for plotting and saving
        base_prompts, base_responses, base_rewards = comparison_results['base_data']
        rlhf_prompts, rlhf_responses, rlhf_rewards = comparison_results['rlhf_data']
        
        # Create plots
        plot_reward_distribution(
            base_rewards,
            "plots/base_model_rewards.png",
            title="Base Model Reward Distribution"
        )
        
        plot_reward_distribution(
            rlhf_rewards,
            "plots/rlhf_model_rewards.png",
            title="RLHF Model Reward Distribution"
        )
        
        # Combined reward distribution plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(base_rewards, bins=30, alpha=0.7, label='Base Model', color='blue')
        plt.hist(rlhf_rewards, bins=30, alpha=0.7, label='RLHF Model', color='red')
        plt.axvline(np.mean(base_rewards), color='blue', linestyle='--', alpha=0.8)
        plt.axvline(np.mean(rlhf_rewards), color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        improvement_per_prompt = np.array(rlhf_rewards) - np.array(base_rewards)
        plt.hist(improvement_per_prompt, bins=30, alpha=0.7, color='green')
        plt.axvline(np.mean(improvement_per_prompt), color='red', linestyle='--')
        plt.xlabel('Reward Improvement')
        plt.ylabel('Frequency')
        plt.title('Per-Sample Reward Improvement')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/reward_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save model outputs
        save_model_outputs(
            prompts=base_prompts,
            responses=base_responses,
            rewards=base_rewards,
            save_path="evaluation_results/base_model_outputs.json",
            model_name="base"
        )
        
        save_model_outputs(
            prompts=rlhf_prompts,
            responses=rlhf_responses,
            rewards=rlhf_rewards,
            save_path="evaluation_results/rlhf_model_outputs.json",
            model_name="rlhf"
        )
        
        # Create side-by-side comparison for sample prompts
        sample_size = min(10, len(eval_prompts))
        compare_models_side_by_side(
            prompts=eval_prompts[:sample_size],
            base_responses=base_responses[:sample_size * 3:3],  # Take first response per prompt
            rlhf_responses=rlhf_responses[:sample_size * 3:3],
            base_rewards=base_rewards[:sample_size * 3:3],
            rlhf_rewards=rlhf_rewards[:sample_size * 3:3],
            save_path="evaluation_results/side_by_side_comparison.json",
            num_examples=sample_size
        )
        
        # Create summary report
        create_summary_report(
            base_metrics=comparison_results['base_results'],
            rlhf_metrics=comparison_results['rlhf_results'],
            training_time=0,  # Not applicable for evaluation
            save_path="evaluation_results/evaluation_summary.json"
        )
        
        # Print key statistics
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Number of prompts evaluated: {len(eval_prompts)}")
        logger.info(f"Samples per prompt: {3}")
        logger.info(f"Total samples: {len(base_rewards)}")
        logger.info("")
        logger.info(f"Base Model Average Reward: {np.mean(base_rewards):.4f} ± {np.std(base_rewards):.4f}")
        logger.info(f"RLHF Model Average Reward: {np.mean(rlhf_rewards):.4f} ± {np.std(rlhf_rewards):.4f}")
        logger.info(f"Average Improvement: {np.mean(rlhf_rewards) - np.mean(base_rewards):.4f}")
        logger.info(f"Improvement Percentage: {((np.mean(rlhf_rewards) - np.mean(base_rewards)) / abs(np.mean(base_rewards))) * 100:.2f}%")
        logger.info("")
        logger.info(f"Percentage of samples improved: {(improvement_per_prompt > 0).mean() * 100:.1f}%")
        logger.info(f"Max improvement: {np.max(improvement_per_prompt):.4f}")
        logger.info(f"Max degradation: {np.min(improvement_per_prompt):.4f}")
        
    else:
        # Only evaluate base model
        logger.info("RLHF model not found, evaluating base model only...")
        base_results, base_prompts, base_responses, base_rewards = evaluator.evaluate_model(
            base_model,
            base_tokenizer,
            eval_prompts,
            num_samples_per_prompt=5,
            max_length=config.experiment.eval_generation_max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        # Log results
        logger.info("Base Model Evaluation Results:")
        for key, value in base_results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save results
        import json
        with open("evaluation_results/base_model_results.json", 'w') as f:
            serializable_results = {
                key: float(val) if isinstance(val, (np.float32, np.float64)) else val
                for key, val in base_results.items()
            }
            json.dump(serializable_results, f, indent=2)
        
        # Create plots
        plot_reward_distribution(
            base_rewards,
            "plots/base_model_rewards.png",
            title="Base Model Reward Distribution"
        )
        
        # Save outputs
        save_model_outputs(
            prompts=base_prompts,
            responses=base_responses,
            rewards=base_rewards,
            save_path="evaluation_results/base_model_outputs.json",
            model_name="base"
        )
        
        logger.info("Evaluation completed. To compare with RLHF model, please run RLHF training first.")
    
    logger.info("\nEvaluation completed successfully!")
    logger.info("Results saved to: evaluation_results/")
    logger.info("Plots saved to: plots/")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate models for Assignment 7")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="Name of the base model"
    )
    parser.add_argument(
        "--rlhf_model",
        type=str,
        default=None,
        help="Path to RLHF model directory"
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default=None,
        help="Path to reward model"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to generate per prompt"
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    if args.base_model:
        config.model.model_name = args.base_model
    if args.rlhf_model:
        config.system.rlhf_model_dir = args.rlhf_model
    if args.reward_model:
        config.system.reward_model_dir = os.path.dirname(args.reward_model)
    if args.max_prompts:
        config.experiment.eval_prompts_sample_size = args.max_prompts
    
    # Validate configuration
    config.validate()
    
    # Run evaluation
    evaluate_models(config)


if __name__ == "__main__":
    main()