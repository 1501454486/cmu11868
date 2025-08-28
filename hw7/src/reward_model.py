"""
Reward Model Implementation for Assignment 7.
This module contains the reward model used in the RLHF pipeline.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Dict, List, Tuple, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewardModelOutput:
    """Output class for reward model predictions."""
    rewards: torch.Tensor
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None


class RewardModel(PreTrainedModel):
    """
    Reward model for RLHF training.
    
    This model takes text inputs and outputs scalar reward values.
    It's based on a pre-trained transformer encoder with a regression head.
    """
    
    def __init__(
        self,
        config,
        model_name: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1
    ):
        """
        Initialize the reward model.
        
        Args:
            config: Model configuration
            model_name: Name of the base model to use
            hidden_size: Hidden size for the reward head
            dropout: Dropout probability
        """
        super().__init__(config)
        
        self.model_name = model_name
        self.hidden_size = hidden_size

        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Output single scalar reward
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the reward head."""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[RewardModelOutput, Tuple]:
        """
        Forward pass of the reward model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for models like BERT)
            position_ids: Position IDs
            head_mask: Head mask
            inputs_embeds: Input embeddings
            return_dict: Whether to return a dict or tuple
            
        Returns:
            RewardModelOutput or tuple containing rewards and hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        
        hidden_states = transformer_outputs.last_hidden_state

        # For BERT-like models, use [CLS] token
        pooled_output = hidden_states[:, 0, :]

        rewards = self.reward_head(pooled_output).squeeze(-1)
        
        if not return_dict:
            return (rewards, hidden_states)
        
        return RewardModelOutput(
            rewards=rewards,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions if hasattr(transformer_outputs, 'attentions') else None,
        )
    
    def get_rewards(
        self, 
        texts: List[str], 
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        batch_size: int = 8
    ) -> List[float]:
        """
        Get reward scores for a list of texts.
        
        Args:
            texts: List of text strings to evaluate
            tokenizer: Tokenizer to use for encoding
            device: Device to run inference on
            batch_size: Batch size for inference
            
        Returns:
            List of reward scores
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        all_rewards = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                outputs = self(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    return_dict=True
                )
                
                batch_rewards = outputs.rewards.cpu().tolist()
                all_rewards.extend(batch_rewards)
        
        return all_rewards


class RewardModelTrainer:
    """
    Trainer for the reward model using preference data.
    """
    
    def __init__(
        self,
        model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_length: int = 512
    ):
        """
        Initialize the reward model trainer.
        
        Args:
            model: The reward model to train
            tokenizer: Tokenizer for text processing
            device: Device to train on
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        # Move model to device
        self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Please refer to:
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
        self.loss_fn = nn.MarginRankingLoss(margin=1.0)
        # END ASSIGN7_1_6
    
    def prepare_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for training.
        
        Args:
            batch: Batch containing 'chosen' and 'rejected' texts
            
        Returns:
            Dictionary with tokenized inputs
        """
        chosen_texts = batch['chosen']
        rejected_texts = batch['rejected']
        
        chosen_encoded = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        rejected_encoded = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        chosen_encoded = {k: v.to(self.device) for k, v in chosen_encoded.items()}
        rejected_encoded = {k: v.to(self.device) for k, v in rejected_encoded.items()}
        
        return {
            'chosen': chosen_encoded,
            'rejected': rejected_encoded
        }
    
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute the ranking loss for a batch.
        
        Args:
            batch: Prepared batch with tokenized inputs
            
        Returns:
            Loss tensor
        """
        # BEGIN ASSIGN7_1
        # TODO: Compute rewards for chosen and rejected responses
        # Get rewards using the model's forward pass
        # Compute ranking loss: chosen should have higher reward than rejected
        # 1. Get rewards for chosen responses
        # 2. Get rewards for rejected responses
        # 3. Compute ranking loss: chosen should have higher reward than rejected
        
        chosen = batch['chosen']
        rejected = batch['rejected']
        chosen_rewards = self.model(**chosen).rewards
        rejected_rewards = self.model(**rejected).rewards
        # y = 1 means that the first input (chosen_rewards) should be ranked higher than the second input
        target = torch.ones_like(chosen_rewards)
        return self.loss_fn(chosen_rewards, rejected_rewards, target), chosen_rewards, rejected_rewards
        # END ASSIGN7_1
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare batch
        prepared_batch = self.prepare_batch(batch)
        
        # Compute loss
        loss, chosen_rewards, rejected_rewards = self.compute_loss(prepared_batch)
        
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_diff = (chosen_rewards - rejected_rewards).mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'reward_diff': reward_diff.item(),
        }
    
    def evaluate_step(self, batch: Dict) -> Dict[str, float]:
        """
        Perform one evaluation step.
        
        Args:
            batch: Evaluation batch
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare batch
            prepared_batch = self.prepare_batch(batch)
            
            # Compute loss
            loss, chosen_rewards, rejected_rewards = self.compute_loss(prepared_batch)
            
            # Compute metrics
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_diff = (chosen_rewards - rejected_rewards).mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'reward_diff': reward_diff.item(),
        }


def create_reward_model(
    model_name: str = "distilbert-base-uncased",
    hidden_size: int = 768,
    dropout: float = 0.1
) -> RewardModel:
    """
    Create a reward model with the specified configuration.
    
    Args:
        model_name: Name of the base transformer model
        hidden_size: Hidden size for the reward head
        dropout: Dropout probability
        
    Returns:
        Initialized RewardModel
    """
    # Load configuration from the base model
    config = AutoConfig.from_pretrained(model_name)
    
    # Create and return the reward model
    model = RewardModel(
        config=config,
        model_name=model_name,
        hidden_size=hidden_size,
        dropout=dropout
    )
    
    logger.info(f"Created reward model based on {model_name}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_reward_model(model_path: str, device: torch.device) -> RewardModel:
    """
    Load a trained reward model from disk.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded RewardModel
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only = False)
    
    # Extract model configuration
    config = checkpoint.get('config')
    model_name = checkpoint.get('model_name', 'distilbert-base-uncased')
    hidden_size = checkpoint.get('hidden_size', 768)
    
    # Create model
    model = create_reward_model(model_name, hidden_size)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Loaded reward model from {model_path}")
    
    return model


def save_reward_model(
    model: RewardModel, 
    model_path: str, 
    tokenizer: Optional[PreTrainedTokenizer] = None,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save a trained reward model to disk.
    
    Args:
        model: The reward model to save
        model_path: Path to save the model
        tokenizer: Optional tokenizer to save with the model
        additional_info: Additional information to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'model_name': model.model_name,
        'hidden_size': model.hidden_size,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, model_path)
    
    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer_path = model_path.replace('.pt', '_tokenizer')
        tokenizer.save_pretrained(tokenizer_path)
    
    logger.info(f"Saved reward model to {model_path}")