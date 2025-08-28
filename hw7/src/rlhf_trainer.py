"""
RLHF Trainer Implementation using VERL Framework for Assignment 7.
This module contains the main RLHF training logic using VERL's PPO implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)

try:
    from verl.trainer import PPOTrainer
    from verl.models import PolicyModel, ValueModel
    from verl.utils import rollout_generator
    from verl.config import PPOConfig
except ImportError:
    logging.warning("VERL not installed. Some functionality may not work.")

from .reward_model import RewardModel
from .config import AssignmentConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    """Data structure for storing rollout results."""
    prompts: List[str]
    responses: List[str]
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring RLHF progress."""
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    reward_mean: float
    reward_std: float
    advantage_mean: float
    advantage_std: float


class VERLPolicyWrapper(PolicyModel):
    """
    Wrapper class to make HuggingFace models compatible with VERL.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize the policy wrapper.
        
        Args:
            model: HuggingFace causal language model
            tokenizer: Tokenizer for the model
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set model to training mode
        self.model.train()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Forward pass through the policy model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs
    
    def generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate responses to prompts and return log probabilities.
        
        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (responses, log_probs)
        """
        self.model.eval()
        
        encoded_prompts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length // 2  # Leave space for generation
        )
        
        device = next(self.model.parameters()).device
        encoded_prompts = {k: v.to(device) for k, v in encoded_prompts.items()}
        
        # Generate responses
        with torch.no_grad():

            generated = self.model.generate(
                input_ids=encoded_prompts['input_ids'],
                attention_mask=encoded_prompts['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )
        
        # Extract generated sequences
        generated_ids = generated.sequences
        
        prompt_lengths = encoded_prompts['input_ids'].shape[1]
        response_ids = generated_ids[:, prompt_lengths:]

        responses = self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Compute log probabilities for generated tokens
        log_probs = self._compute_log_probs(generated_ids, generated.scores)
        
        self.model.train()
        return responses, log_probs
    
    def _compute_log_probs(
        self, 
        generated_ids: torch.Tensor, 
        scores: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute log probabilities for generated sequences.
        
        Args:
            generated_ids: Generated token IDs
            scores: Generation scores from model
            
        Returns:
            Log probabilities tensor
        """
        if not scores:
            # Fallback: compute log probs through forward pass
            return self._compute_log_probs_fallback(generated_ids)
        
        log_probs = torch.stack([
            torch.log_softmax(score, dim=-1) 
            for score in scores
        ], dim=1)
        
        # Extract log probs for generated tokens
        batch_size, seq_len = generated_ids.shape
        generated_log_probs = []
        
        for i in range(batch_size):
            sequence_log_probs = []
            for j in range(1, seq_len):  # Skip first token (from prompt)
                if j-1 < log_probs.shape[1]:
                    token_id = generated_ids[i, j]
                    token_log_prob = log_probs[i, j-1, token_id]
                    sequence_log_probs.append(token_log_prob)
            
            if sequence_log_probs:
                generated_log_probs.append(torch.stack(sequence_log_probs).sum())
            else:
                generated_log_probs.append(torch.tensor(0.0, device=generated_ids.device))
        
        return torch.stack(generated_log_probs)
    
    def _compute_log_probs_fallback(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Fallback method to compute log probabilities."""
        with torch.no_grad():
            outputs = self.model(generated_ids)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            
            # Sum log probs for each sequence
            sequence_log_probs = []
            for i in range(generated_ids.shape[0]):
                seq_log_prob = 0.0
                for j in range(generated_ids.shape[1] - 1):
                    token_id = generated_ids[i, j + 1]
                    seq_log_prob += log_probs[i, j, token_id]
                sequence_log_probs.append(seq_log_prob)
            
            return torch.stack(sequence_log_probs)


class VERLValueWrapper(ValueModel):
    """
    Value model wrapper for VERL integration.
    """
    
    def __init__(self, policy_model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize value model based on policy model.
        
        Args:
            policy_model: Base policy model
            tokenizer: Tokenizer for the model
        """
        super().__init__()
        self.tokenizer = tokenizer
        
        # Create value head on top of the policy model
        self.backbone = policy_model
        
        self.value_head = nn.Linear(policy_model.config.hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Forward pass to compute state values."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Use last hidden state for value prediction
        last_hidden_state = outputs.hidden_states[-1]
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_hidden = sum_hidden / sum_mask
        else:
            pooled_hidden = last_hidden_state.mean(dim=1)
        
        # Compute value
        values = self.value_head(pooled_hidden).squeeze(-1)
        
        return values


class VERLTrainer:
    """
    Main RLHF trainer using VERL framework.
    """
    
    def __init__(
        self,
        policy_model: PreTrainedModel,
        reward_model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        config: AssignmentConfig,
        device: torch.device
    ):
        """
        Initialize VERL RLHF trainer.
        
        Args:
            policy_model: Policy model to train
            reward_model: Trained reward model
            tokenizer: Tokenizer for text processing
            config: Training configuration
            device: Training device
        """
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        # Wrap models for VERL compatibility
        self.policy = VERLPolicyWrapper(policy_model, tokenizer)
        self.value_model = VERLValueWrapper(policy_model, tokenizer)
        self.reward_model = reward_model
        
        # Move models to device
        self.policy.to(device)
        self.value_model.to(device)
        self.reward_model.to(device)
        
        # Set reward model to eval mode
        self.reward_model.eval()
        
        # Create PPO configuration
        self.ppo_config = self._create_ppo_config()
        
        # Initialize VERL PPO trainer
        try:
            self.ppo_trainer = PPOTrainer(
                policy_model=self.policy,
                value_model=self.value_model,
                config=self.ppo_config
            )
        except Exception as e:
            logger.warning(f"Failed to initialize VERL PPOTrainer: {e}")
            logger.warning("Falling back to custom PPO implementation")
            self.ppo_trainer = None
            self._init_custom_optimizers()
    
    def _create_ppo_config(self) -> Dict[str, Any]:
        """Create PPO configuration for VERL."""
        return {
            'learning_rate': self.config.training.ppo_learning_rate,
            'batch_size': self.config.verl.train_batch_size,
            'mini_batch_size': self.config.verl.train_mini_batch_size,
            'ppo_epochs': self.config.verl.ppo_epochs,
            'clip_eps': self.config.verl.ppo_clip_eps,
            'target_kl': self.config.verl.ppo_target_kl,
            'gamma': self.config.training.ppo_gamma,
            'gae_lambda': self.config.training.ppo_gae_lambda,
            'value_coef': self.config.training.ppo_value_coef,
            'entropy_coef': self.config.training.ppo_entropy_coef,
            'max_grad_norm': self.config.training.ppo_max_grad_norm,
        }
    
    def _init_custom_optimizers(self):
        """Initialize custom optimizers for fallback implementation."""

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.training.ppo_learning_rate,
            weight_decay=0.01
        )
        
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=self.config.training.ppo_learning_rate * 2,  # Higher LR for value
            weight_decay=0.01
        )
    
    def generate_rollouts(self, prompts: List[str]) -> RolloutBatch:
        """
        Generate rollouts using the current policy.
        
        Args:
            prompts: List of prompts to generate responses for
            
        Returns:
            RolloutBatch containing rollout data
        """
        # Generate responses
        responses, log_probs = self.policy.generate(
            prompts=prompts,
            max_length=self.config.verl.rollout_max_length,
            temperature=self.config.verl.rollout_temperature,
            top_p=self.config.verl.rollout_top_p,
            do_sample=True
        )
        
        full_texts = [f"{prompt} {response}" for prompt, response in zip(prompts, responses)]
        rewards = torch.tensor(
            self.reward_model.get_rewards(full_texts, self.tokenizer, self.device),
            device=self.device,
            dtype=torch.float32
        )
        
        # Apply reward clipping if configured
        if hasattr(self.config.verl, 'reward_clip') and self.config.verl.reward_clip > 0:
            rewards = torch.clamp(rewards, -self.config.verl.reward_clip, self.config.verl.reward_clip)
        
        # Normalize rewards if configured
        if getattr(self.config.verl, 'reward_normalize', False):
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get values from value model
        full_inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.verl.rollout_max_length,
            return_tensors="pt"
        )
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
        
        with torch.no_grad():
            values = self.value_model(
                input_ids=full_inputs['input_ids'],
                attention_mask=full_inputs['attention_mask']
            )
        
        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(rewards, values)
        
        return RolloutBatch(
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns
        )
    
    def _compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward values
            values: State values
            
        Returns:
            Tuple of (advantages, returns)
        """
        # BEGIN ASSIGN7_2_1
        # TODO: Implement simplified GAE computation
        # For this assignment, treat each response as a single-step episode:
        # 1. Compute returns = rewards + gamma * values (assume next state value = current value)
        # 2. Compute advantages = returns - values
        # 3. Normalize advantages: (advantages - mean) / (std + 1e-8)
        returns = rewards + self.ppo_config['gamma'] * values
        advantages = returns - values
        mean = advantages.mean()
        std = advantages.std()
        advantages = (advantages - mean) / (std + 1e-8)
        
        # END ASSIGN7_2_1
        
        return advantages, returns
    
    def train_step(self, rollout_batch: RolloutBatch) -> TrainingMetrics:
        """
        Perform one PPO training step.
        
        Args:
            rollout_batch: Batch of rollout data
            
        Returns:
            Training metrics
        """
        if self.ppo_trainer is not None:
            # Use VERL's PPO trainer
            return self._train_step_verl(rollout_batch)
        else:
            # Use custom PPO implementation
            return self._train_step_custom(rollout_batch)
    
    def _train_step_verl(self, rollout_batch: RolloutBatch) -> TrainingMetrics:
        """Training step using VERL's PPO trainer."""
        # Convert rollout batch to VERL format
        verl_batch = {
            'observations': rollout_batch.prompts,
            'actions': rollout_batch.responses,
            'rewards': rollout_batch.rewards,
            'old_log_probs': rollout_batch.log_probs,
            'values': rollout_batch.values,
            'advantages': rollout_batch.advantages,
            'returns': rollout_batch.returns
        }
        
        # Train using VERL's PPO trainer
        metrics = self.ppo_trainer.train_step(verl_batch)
        
        return TrainingMetrics(
            policy_loss=metrics.get('policy_loss', 0.0),
            value_loss=metrics.get('value_loss', 0.0),
            entropy=metrics.get('entropy', 0.0),
            kl_divergence=metrics.get('kl_divergence', 0.0),
            reward_mean=rollout_batch.rewards.mean().item(),
            reward_std=rollout_batch.rewards.std().item(),
            advantage_mean=rollout_batch.advantages.mean().item(),
            advantage_std=rollout_batch.advantages.std().item()
        )
    
    def _train_step_custom(self, rollout_batch: RolloutBatch) -> TrainingMetrics:
        """Custom PPO training step implementation."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        
        # Prepare full texts for training
        full_texts = [f"{prompt} {response}" for prompt, response 
                     in zip(rollout_batch.prompts, rollout_batch.responses)]
        
        # Tokenize full texts
        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.verl.rollout_max_length,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # PPO training epochs
        for epoch in range(self.config.verl.ppo_epochs):
            # Forward pass through policy
            policy_outputs = self.policy(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            
            # Compute new log probabilities
            log_probs = torch.log_softmax(policy_outputs.logits, dim=-1)
            new_log_probs = self._extract_action_log_probs(
                log_probs, encoded['input_ids']
            )
            
            # BEGIN ASSIGN7_2_2
            # TODO: Compute PPO loss
            # 1. Compute probability ratio: exp(new_log_probs - old_log_probs)
            # 2. Compute surrogate losses:
            #    - surr1 = ratio * advantages
            #    - surr2 = clipped_ratio * advantages (clip ratio between 1-eps and 1+eps)
            # 3. Policy loss = -min(surr1, surr2).mean()
            # 4. Compute entropy bonus from policy logits
            raise NotImplementedError("Need to implement PPO loss computation for Assignment 7")
            
            # END ASSIGN7_2_2
            
            # Total policy loss with entropy bonus
            total_policy_loss_step = (
                policy_loss - 
                self.config.training.ppo_entropy_coef * entropy
            )
            
            # Value function training
            values = self.value_model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            
            value_loss = nn.MSELoss()(values, rollout_batch.returns)
            
            # Compute KL divergence for monitoring
            kl_div = (rollout_batch.log_probs - new_log_probs).mean()
            
            # Backward pass for policy
            self.policy_optimizer.zero_grad()
            total_policy_loss_step.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.config.training.ppo_max_grad_norm
            )
            self.policy_optimizer.step()
            
            # Backward pass for value function
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(),
                self.config.training.ppo_max_grad_norm
            )
            self.value_optimizer.step()
            
            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl_div.item()
            
            # Early stopping if KL divergence is too large
            if kl_div.item() > self.config.verl.ppo_target_kl * 2:
                logger.warning(f"Early stopping due to large KL divergence: {kl_div.item()}")
                break
        
        num_epochs = self.config.verl.ppo_epochs
        return TrainingMetrics(
            policy_loss=total_policy_loss / num_epochs,
            value_loss=total_value_loss / num_epochs,
            entropy=total_entropy / num_epochs,
            kl_divergence=total_kl / num_epochs,
            reward_mean=rollout_batch.rewards.mean().item(),
            reward_std=rollout_batch.rewards.std().item(),
            advantage_mean=rollout_batch.advantages.mean().item(),
            advantage_std=rollout_batch.advantages.std().item()
        )
    
    def _extract_action_log_probs(
        self, 
        log_probs: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract log probabilities for the action tokens.
        
        Args:
            log_probs: Log probabilities from model output
            input_ids: Input token IDs
            
        Returns:
            Log probabilities for action tokens
        """
        batch_size, seq_len = input_ids.shape
        action_log_probs = []
        
        for i in range(batch_size):
            seq_log_prob = 0.0
            for j in range(seq_len - 1):
                token_id = input_ids[i, j + 1]
                seq_log_prob += log_probs[i, j, token_id]
            action_log_probs.append(seq_log_prob)
        
        return torch.stack(action_log_probs)
    
    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the policy distribution.
        
        Args:
            logits: Model logits
            
        Returns:
            Entropy values
        """
        # BEGIN ASSIGN7_2_3
        # TODO: Compute entropy from logits
        # 1. Convert logits to probabilities using softmax
        # 2. Convert logits to log_probabilities using log_softmax
        # 3. Compute entropy: -(probs * log_probs).sum(dim=-1)
        # 4. Return mean over sequence length
        raise NotImplementedError("Need to implement entropy computation for Assignment 7")
        
        # END ASSIGN7_2_3
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Training metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.model.state_dict(),
            'value_state_dict': self.value_model.state_dict(),
            'policy_optimizer_state_dict': getattr(self, 'policy_optimizer', {}).state_dict() if hasattr(self, 'policy_optimizer') else {},
            'value_optimizer_state_dict': getattr(self, 'value_optimizer', {}).state_dict() if hasattr(self, 'value_optimizer') else {},
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.policy.model.load_state_dict(checkpoint['policy_state_dict'])
        self.value_model.load_state_dict(checkpoint['value_state_dict'])
        
        # Load optimizer states if available
        if hasattr(self, 'policy_optimizer') and 'policy_optimizer_state_dict' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if hasattr(self, 'value_optimizer') and 'value_optimizer_state_dict' in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def create_rlhf_trainer(
    model_name: str,
    reward_model: RewardModel,
    config: AssignmentConfig,
    device: torch.device
) -> VERLTrainer:
    """
    Create and initialize RLHF trainer.
    
    Args:
        model_name: Name of the base language model
        reward_model: Trained reward model
        config: Training configuration
        device: Training device
        
    Returns:
        Initialized VERLTrainer
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Create trainer
    trainer = VERLTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    
    logger.info(f"Created RLHF trainer with model: {model_name}")
    logger.info(f"Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    
    return trainer


def evaluate_policy(
    trainer: VERLTrainer,
    eval_prompts: List[str],
    num_samples: int = 5
) -> Dict[str, float]:
    """
    Evaluate the policy on a set of prompts.
    
    Args:
        trainer: RLHF trainer with trained policy
        eval_prompts: List of evaluation prompts
        num_samples: Number of samples to generate per prompt
        
    Returns:
        Evaluation metrics
    """
    trainer.policy.model.eval()
    
    all_rewards = []
    all_response_lengths = []
    
    with torch.no_grad():
        for prompt in eval_prompts:
            # Generate multiple samples per prompt
            responses, _ = trainer.policy.generate(
                prompts=[prompt] * num_samples,
                max_length=trainer.config.verl.rollout_max_length,
                temperature=trainer.config.verl.rollout_temperature,
                top_p=trainer.config.verl.rollout_top_p,
                do_sample=True
            )
            
            # Get rewards for generated responses
            full_texts = [f"{prompt} {response}" for response in responses]
            rewards = trainer.reward_model.get_rewards(
                full_texts, 
                trainer.tokenizer, 
                trainer.device
            )
            
            all_rewards.extend(rewards)
            all_response_lengths.extend([len(response.split()) for response in responses])
    
    trainer.policy.model.train()
    
    return {
        'mean_reward': sum(all_rewards) / len(all_rewards),
        'std_reward': torch.tensor(all_rewards).std().item(),
        'min_reward': min(all_rewards),
        'max_reward': max(all_rewards),
        'mean_response_length': sum(all_response_lengths) / len(all_response_lengths),
        'num_evaluations': len(all_rewards)
    }