"""
Configuration file for Assignment 7 - VERL RLHF Training
This file contains all configuration parameters for the RLHF pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for the language model and reward model."""
    
    # Base model settings
    model_name: str = "gpt2"  # Base model to fine-tune
    model_revision: str = "main"
    trust_remote_code: bool = False
    
    # Tokenizer settings
    tokenizer_name: Optional[str] = None  # Use same as model_name if None
    max_length: int = 512
    pad_token_id: Optional[int] = None  # Will be set automatically
    
    # Reward model settings
    reward_model_name: str = "distilbert-base-uncased"
    reward_model_hidden_size: int = 768
    reward_model_dropout: float = 0.1


@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    
    # Data paths
    train_prompts_path: str = "data/train_prompts.json"
    eval_prompts_path: str = "data/eval_prompts.json"
    preference_data_path: str = "data/preference_data.json"
    
    # Data processing
    max_prompt_length: int = 256
    max_response_length: int = 256
    train_batch_size: int = 4
    eval_batch_size: int = 8
    
    # Data splits
    train_split_ratio: float = 0.8
    validation_split_ratio: float = 0.1
    test_split_ratio: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Reward model training
    reward_learning_rate: float = 2e-5
    reward_num_epochs: int = 3
    reward_warmup_steps: int = 100
    reward_weight_decay: float = 0.01
    
    # RLHF/PPO training
    ppo_learning_rate: float = 1e-6
    ppo_num_epochs: int = 2
    ppo_batch_size: int = 4
    ppo_mini_batch_size: int = 2
    
    # PPO specific parameters
    ppo_clip_range: float = 0.2
    ppo_clip_range_vf: float = 0.2
    ppo_gamma: float = 1.0
    ppo_gae_lambda: float = 0.95
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 1.0
    
    # Generation parameters for rollouts
    generation_max_length: int = 256
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_do_sample: bool = True
    generation_num_return_sequences: int = 1


@dataclass
class VERLConfig:
    """VERL-specific configuration parameters."""
    
    # Rollout settings
    rollout_batch_size: int = 4
    rollout_max_length: int = 256
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.9
    
    # Training settings
    train_batch_size: int = 2
    train_mini_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    
    # PPO algorithm settings
    ppo_epochs: int = 4
    ppo_clip_eps: float = 0.2
    ppo_target_kl: float = 0.01
    ppo_adaptive_kl: bool = True
    ppo_kl_penalty: float = 0.1
    
    # Value function settings
    use_value_head: bool = True
    value_loss_coef: float = 0.5
    
    # Reward settings
    reward_clip: float = 5.0
    reward_normalize: bool = True


@dataclass
class SystemConfig:
    """System and hardware configuration."""
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "fp16"  # Options: "no", "fp16", "bf16"
    
    # Distributed training
    world_size: int = 1
    local_rank: int = 0
    
    # Memory and performance
    dataloader_num_workers: int = 2
    pin_memory: bool = True
    gradient_checkpointing: bool = False
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Output directories
    output_dir: str = "outputs"
    reward_model_dir: str = "outputs/reward_model"
    rlhf_model_dir: str = "outputs/rlhf_model"
    logs_dir: str = "logs"


@dataclass
class ExperimentConfig:
    """Experiment tracking and evaluation configuration."""
    
    # Experiment tracking
    wandb_project: str = "llmsys_assignment7"
    wandb_entity: Optional[str] = None
    experiment_name: str = "verl_rlhf_baseline"
    
    # Evaluation settings
    eval_prompts_sample_size: int = 100
    eval_generation_max_length: int = 256
    eval_batch_size: int = 8
    
    # Metrics to track
    track_reward_distribution: bool = True
    track_kl_divergence: bool = True
    track_response_length: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False


@dataclass
class AssignmentConfig:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    verl: VERLConfig = VERLConfig()
    system: SystemConfig = SystemConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "verl": self.verl.__dict__,
            "system": self.system.__dict__,
            "experiment": self.experiment.__dict__,
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section_obj = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Check data split ratios sum to 1.0
        total_ratio = (
            self.data.train_split_ratio + 
            self.data.validation_split_ratio + 
            self.data.test_split_ratio
        )
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        # Check batch size compatibility
        if self.training.ppo_batch_size % self.training.ppo_mini_batch_size != 0:
            raise ValueError(
                "ppo_batch_size must be divisible by ppo_mini_batch_size"
            )
        
        # Check VERL batch size compatibility
        if self.verl.train_batch_size % self.verl.train_mini_batch_size != 0:
            raise ValueError(
                "VERL train_batch_size must be divisible by train_mini_batch_size"
            )
        
        # Validate clip ranges
        if not (0 < self.training.ppo_clip_range < 1):
            raise ValueError("ppo_clip_range must be between 0 and 1")
        
        if not (0 < self.verl.ppo_clip_eps < 1):
            raise ValueError("ppo_clip_eps must be between 0 and 1")


# Default configuration instance
default_config = AssignmentConfig()


def get_config() -> AssignmentConfig:
    """
    Get the default configuration.
    
    Returns:
        AssignmentConfig: Default configuration instance
    """
    return default_config


def load_config_from_file(config_path: str) -> AssignmentConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        AssignmentConfig: Loaded configuration
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = AssignmentConfig()
    config.update_from_dict(config_dict)
    config.validate()
    
    return config


def save_config_to_file(config: AssignmentConfig, config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration to save
        config_path: Path to save the configuration file
    """
    import yaml
    
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)