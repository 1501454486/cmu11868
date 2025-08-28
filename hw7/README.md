# Assignment 7: Introduction to VERL Framework

**CMU 11868 LLM Systems Spring 2025**


## Overview

In this assignment, you will learn to use VERL (Volcano Engine Reinforcement Learning), a flexible and efficient reinforcement learning framework designed for large language models. You'll implement a basic RLHF (Reinforcement Learning from Human Feedback) pipeline using VERL to fine-tune a small language model for harmless and helpful responses.

## Learning Objectives

After completing this assignment, you will be able to:
- Understand the basic concepts of RLHF and its application to LLMs
- Set up and configure the VERL framework
- Implement a simple reward model for evaluating LLM outputs
- Use VERL's PPO implementation to fine-tune a language model
- Evaluate the performance improvements after RLHF training

## Background Reading

Before starting the implementation, familiarize yourself with the core concepts:

**Essential Reading:**
- [VERL Documentation](https://verl.readthedocs.io/en/latest/) - Framework overview and API reference
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256) - The research foundation behind VERL
- [RLHF Tutorial](https://huggingface.co/blog/rlhf) - Comprehensive introduction to RLHF methodology
- [PPO Algorithm Explained](https://openai.com/research/openai-baselines-ppo) - Understanding the RL algorithm

**Key Concepts:**
- **RLHF Pipeline**: Human preference data → Reward model training → Policy optimization with PPO
- **VERL's Hybrid Architecture**: Separation of generation (inference) and training phases for scalability
- **PPO Algorithm**: Policy gradient method with clipping for stable training
- **Reward Model**: Neural network trained to predict human preferences

## Environment Setup

### Step 1: Clone the Assignment Repository

```bash
git clone https://github.com/llmsystem/llmsys_f25_hw7.git
cd llmsys_f25_hw7
```

### Step 2: Create a Virtual Environment

```bash
conda create -n llmsys_hw7 python=3.9
conda activate llmsys_hw7
```

### Step 3: Install Dependencies

```bash

# Install VERL and other dependencies
pip install -r requirements.txt
```

## Training Data

### Data Sources

This assignment uses the **Anthropic/hh-rlhf** dataset from Hugging Face, which contains real human preference data for helpfulness and harmlessness. This is the same dataset used in many RLHF research papers.

### Data Loading and Preparation

The training data is automatically downloaded and prepared when you first run the training scripts. You can also prepare it manually:

```bash
# Download and prepare the dataset
python scripts/prepare_data.py --dataset Anthropic/hh-rlhf --output_dir data --max_samples 10000
```

## Problems

### Problem 1: Implementing a Reward Model (40 points)

**1.1** Complete the loss implementation in `src/reward_model.py`:

In this problem, you 

- Implement the ranking loss for the reward training in `compute_loss`.
- The model is based on a pre-trained transformer (DistilBERT)

**1.2** Train your reward model using the provided preference data:

```bash
python scripts/train_reward_model.py
```

### Problem 2: RLHF Training with VERL (40 points)

**2.1** Complete the RLHF trainer implementation in `src/rlhf_trainer.py`:

- Implement the `VERLTrainer` class using VERL's PPO implementation

**2.2** Run the RLHF training process:

```bash
python scripts/run_rlhf.py --model_name gpt2 --config src/config.py
```

### Problem 3: Evaluation and Analysis (20 points)

**3.1** Run comprehensive evaluation:

```bash
python scripts/evaluate.py --base_model gpt2 --rlhf_model outputs/rlhf_model --config src/config.py
```

### Testing

Run the provided tests to verify your implementation:

```bash
python -m pytest tests/ -v
```

## Submission Instructions

1. **Code Submission**: 
   - Ensure all code runs without errors
   - Include all required output files
   - Test your implementation with `python -m pytest tests/`

2. **Create Submission Archive**:
   ```bash
   # Remove large model files but keep small checkpoints
   find outputs/ -name "*.bin" -size +100M -delete
   
   # Create submission zip
   zip -r assignment7_[your_andrew_id].zip . -x "*.git*" "*__pycache__*" "*.pyc"
   ```
