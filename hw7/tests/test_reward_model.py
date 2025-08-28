"""
Test cases for reward model implementation.
"""

import sys
import os
import torch
import pytest
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from reward_model import create_reward_model, RewardModelTrainer, save_reward_model, load_reward_model
from config import get_config


class TestRewardModel:
    """Test cases for RewardModel class."""
    
    @pytest.fixture
    def model_and_tokenizer(self):
        """Create a reward model and tokenizer for testing."""
        model = create_reward_model(
            model_name="distilbert-base-uncased",
            hidden_size=256,
            dropout=0.1
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    
    def test_model_creation(self, model_and_tokenizer):
        """Test that the reward model is created correctly."""
        model, tokenizer = model_and_tokenizer
        
        assert model is not None
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'reward_head')
        assert model.model_name == "distilbert-base-uncased"
        assert model.hidden_size == 256
    
    def test_forward_pass(self, model_and_tokenizer):
        """Test forward pass through the model."""
        model, tokenizer = model_and_tokenizer
        
        # Create sample input
        texts = ["This is a good response.", "This is a bad response."]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Forward pass
        outputs = model(**encoded)
        
        assert hasattr(outputs, 'rewards')
        assert outputs.rewards.shape == (2,)  # Batch size of 2
        assert torch.all(torch.isfinite(outputs.rewards))
    
    def test_get_rewards(self, model_and_tokenizer):
        """Test the get_rewards method."""
        model, tokenizer = model_and_tokenizer
        
        texts = [
            "This is a helpful response.",
            "This response is not very good.",
            "Another test response."
        ]
        
        rewards = model.get_rewards(texts, tokenizer, batch_size=2)
        
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        assert all(torch.isfinite(torch.tensor(r)) for r in rewards)
    
    def test_reward_model_trainer(self, model_and_tokenizer):
        """Test the RewardModelTrainer class."""
        model, tokenizer = model_and_tokenizer
        device = torch.device("cpu")
        
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            learning_rate=1e-4,
            max_length=128
        )
        
        assert trainer.model == model
        assert trainer.tokenizer == tokenizer
        assert trainer.device == device
    
    def test_training_step(self, model_and_tokenizer):
        """Test a single training step."""
        model, tokenizer = model_and_tokenizer
        device = torch.device("cpu")
        
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            learning_rate=1e-4,
            max_length=128
        )
        
        # Create sample batch
        batch = {
            'chosen': ["This is a good response."],
            'rejected': ["This is a bad response."]
        }
        
        # Training step
        metrics = trainer.train_step(batch)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'chosen_reward_mean' in metrics
        assert 'rejected_reward_mean' in metrics
        assert isinstance(metrics['loss'], float)
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluation_step(self, model_and_tokenizer):
        """Test a single evaluation step."""
        model, tokenizer = model_and_tokenizer
        device = torch.device("cpu")
        
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            learning_rate=1e-4,
            max_length=128
        )
        
        # Create sample batch
        batch = {
            'chosen': ["This is a good response.", "Another good response."],
            'rejected': ["This is a bad response.", "Another bad response."]
        }
        
        # Evaluation step
        metrics = trainer.evaluate_step(batch)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'reward_diff' in metrics
        assert isinstance(metrics['loss'], float)
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_save_and_load_model(self, model_and_tokenizer, tmp_path):
        """Test saving and loading the reward model."""
        model, tokenizer = model_and_tokenizer
        
        # Save model
        model_path = tmp_path / "test_reward_model.pt"
        save_reward_model(
            model=model,
            model_path=str(model_path),
            tokenizer=tokenizer,
            additional_info={'test_key': 'test_value'}
        )
        
        assert model_path.exists()
        
        # Load model
        loaded_model = load_reward_model(str(model_path), torch.device("cpu"))
        
        assert loaded_model is not None
        assert loaded_model.model_name == model.model_name
        assert loaded_model.hidden_size == model.hidden_size
        
        # Test that loaded model produces same outputs
        texts = ["Test input"]
        original_rewards = model.get_rewards(texts, tokenizer)
        loaded_rewards = loaded_model.get_rewards(texts, tokenizer)
        
        # Should be very close (small numerical differences expected)
        assert abs(original_rewards[0] - loaded_rewards[0]) < 1e-4


class TestRewardModelIntegration:
    """Integration tests for reward model."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training with minimal data."""
        # Create model and tokenizer
        model = create_reward_model(
            model_name="distilbert-base-uncased",
            hidden_size=128,
            dropout=0.1
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device("cpu")
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            learning_rate=1e-3,  # Higher LR for faster convergence in test
            max_length=64
        )
        
        # Create minimal training data
        training_batches = [
            {
                'chosen': ["This is helpful and good."],
                'rejected': ["This is not helpful."]
            },
            {
                'chosen': ["Another good response."],
                'rejected': ["Another bad response."]
            }
        ]
        
        # Train for a few steps
        initial_loss = None
        for i, batch in enumerate(training_batches * 2):  # 4 total steps
            metrics = trainer.train_step(batch)
            
            if i == 0:
                initial_loss = metrics['loss']
            
            # Check that metrics are reasonable
            assert metrics['loss'] > 0
            assert 0 <= metrics['accuracy'] <= 1
            assert isinstance(metrics['chosen_reward_mean'], float)
            assert isinstance(metrics['rejected_reward_mean'], float)
        
        # Loss should generally decrease (though not guaranteed in such a short training)
        final_metrics = trainer.train_step(training_batches[0])
        
        # At minimum, check that training doesn't crash and produces valid outputs
        assert final_metrics['loss'] > 0
        assert 0 <= final_metrics['accuracy'] <= 1
    
    def test_batch_size_handling(self):
        """Test that the model handles different batch sizes correctly."""
        model = create_reward_model(
            model_name="distilbert-base-uncased",
            hidden_size=64,
            dropout=0.1
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            texts = [f"Test text {i}" for i in range(batch_size)]
            rewards = model.get_rewards(texts, tokenizer, batch_size=2)
            
            assert len(rewards) == batch_size
            assert all(isinstance(r, float) for r in rewards)
    
    def test_different_text_lengths(self):
        """Test model with texts of different lengths."""
        model = create_reward_model(
            model_name="distilbert-base-uncased",
            hidden_size=64,
            dropout=0.1
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        texts = [
            "Short.",
            "This is a medium length text with several words.",
            "This is a much longer text that contains many more words and should test the model's ability to handle longer sequences. It includes multiple sentences and covers various topics to ensure robust testing."
        ]
        
        rewards = model.get_rewards(texts, tokenizer)
        
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        assert all(torch.isfinite(torch.tensor(r)) for r in rewards)


if __name__ == "__main__":
    pytest.main([__file__])