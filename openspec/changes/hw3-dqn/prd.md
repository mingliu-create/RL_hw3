# Homework 3: DQN and its variants

## 1. Objective
Implement and compare various Deep Q-Network (DRL) algorithms based on the provided reference code, focusing on static mode, player mode, and random mode gridworld environments.

## 2. Reference Repository
- **Base repo**: DRL in Action (English) GitHub repo: [DeepReinforcementLearningInAction](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)
- Use the updated starter code provided by the instructor as the baseline.

## 3. Scope of Work

### 3.1. HW3-1: Naive DQN for static mode (30%)
- **Implementation**: Run the provided code with naive DQN or Experience buffer replay.
- **Task**: 
  - Basic DQN implementation for an easy environment.
  - Implementation of Experience Replay Buffer.
- **Reporting**: Chat with ChatGPT (or Gemini) to clarify the code understanding and write a short understanding report.

### 3.2. HW3-2: Enhanced DQN Variants for player mode (40%)
- **Implementation**: Implement and compare the following DQN variants:
  - Double DQN
  - Dueling DQN
- **Focus**: Emphasize and document how these variants improve upon the basic DQN approach.

### 3.3. HW3-3: Enhance DQN for random mode WITH Training Tips (30%)
- **Implementation**: Convert the PyTorch DQN model to either **Keras** or **PyTorch Lightning**.
- **Enhancements (Bonus points)**: Integrate training techniques to stabilize and improve learning:
  - Gradient clipping
  - Learning rate scheduling
  - Other applicable techniques.

### 3.4. HW3-4: Rainbow DQN (Bonus)
- **Implementation**: Use Rainbow DQN to solve the Random Mode GridWorld.
- **Workflow**: First analyze the problem and formulate a strategy, then implement the Rainbow DQN approach.

## 4. Delivery
- The completed codebase with all implemented models.
- The understanding report for HW3-1.
- Comparisons and discussions for HW3-2.
- A well-commented and structured implementation suitable for grading.
