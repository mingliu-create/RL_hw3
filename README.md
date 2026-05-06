# Reinforcement Learning - Homework 3: DQN and its Variants

This repository contains the implementation for Homework 3 of the Reinforcement Learning course. The project involves building and comparing various Deep Q-Network (DQN) architectures in different configurations of a GridWorld environment.

## 📁 Project Structure

The underlying `Gridworld` environment was adapted from the [DeepReinforcementLearningInAction](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction) repository. It operates in three main configurations:
- **`static`**: The player and all objects are in fixed positions.
- **`player`**: The player starts in a random position, while other objects are fixed.
- **`random`**: Everything (Player, Goal, Pit, Wall) is randomized every episode.

The project is split into four main Python scripts corresponding to the homework requirements:

### 1. HW3-1: Naive DQN (`hw3_1_naive_dqn.py`)
- **Environment Mode**: `static`
- **Algorithm**: Basic Deep Q-Network with an Experience Replay Buffer.
- **Key Feature**: Stores past transitions and samples mini-batches during training to break temporal correlation and stabilize learning.

### 2. HW3-2: Enhanced DQN Variants (`hw3_2_double_dueling_dqn.py`)
- **Environment Mode**: `player`
- **Algorithms**: Double DQN and Dueling DQN.
- **Key Features**: 
  - **Double DQN**: Mitigates the overestimation of Q-values by decoupling action selection (via policy net) from action evaluation (via target net).
  - **Dueling DQN**: Splits the network into Value and Advantage streams, allowing the model to learn the fundamental value of a state independent of the specific action taken.

### 3. HW3-3: PyTorch Lightning DQN with Tricks (`hw3_3_lightning_dqn.py`)
- **Environment Mode**: `random`
- **Framework**: PyTorch Lightning
- **Key Features**: Converted standard PyTorch implementation into the Lightning framework for better scalability. Integrated specific training tips:
  - **Gradient Clipping** (`gradient_clip_val=1.0`): Prevents exploding gradients during unstable training phases.
  - **Learning Rate Scheduling** (`StepLR`): Progressively reduces learning rate for smoother convergence.

### 4. HW3-4: Rainbow Lite DQN [Bonus] (`hw3_4_rainbow_dqn.py`)
- **Environment Mode**: `random`
- **Algorithm**: Simplified Rainbow DQN
- **Key Features**: Combines Double DQN, Dueling DQN, and N-Step (Multi-step) Returns. N-step returns help rapidly propagate delayed rewards (like hitting a +10 goal) back through the trajectory, which is crucial for solving highly randomized environments.

## 📝 Reports
Please view the `Report.md` file for an in-depth understanding report, architectural comparisons, and an analysis of the Rainbow DQN approach.

## 🚀 How to Run

1. Make sure you have `torch`, `pytorch-lightning`, and `numpy` installed.
2. Run any of the scripts directly from the command line:

```bash
python hw3_1_naive_dqn.py
python hw3_2_double_dueling_dqn.py
python hw3_3_lightning_dqn.py
python hw3_4_rainbow_dqn.py
```
