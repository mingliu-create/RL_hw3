# Homework 3: DQN and its Variants - Understanding Report

## HW3-1: Naive DQN for Static Mode
**Understanding Report:**
In the static mode, the environment always initializes with fixed positions for the player, goal, pit, and wall. The Naive DQN aims to find the optimal policy in this deterministic setup. 
The implementation uses a simple Neural Network with PyTorch to approximate the Q-function. We added an **Experience Replay Buffer** which stores transitions `(state, action, reward, next_state, done)`. During training, random mini-batches of size 32 are sampled from this buffer. This breaks the temporal correlation between sequential experiences and stabilizes the learning process. The loss is calculated using the Mean Squared Error between the predicted Q-values and the target Q-values computed via the Bellman equation. For exploration, an epsilon-greedy strategy is applied, with epsilon decaying progressively.

## HW3-2: Enhanced DQN Variants for Player Mode
**Comparisons & Discussions:**
In player mode, the player's initial position is randomized, requiring the model to generalize better across different starting states. To tackle this, we implemented two enhancements over Naive DQN:

1. **Double DQN**:
   Standard DQN often overestimates Q-values because the `max` operation is used both to select and evaluate actions. Double DQN decouples this by using the `policy_net` to select the best action for the next state, and the `target_net` to evaluate the Q-value for that chosen action. This prevents the overestimation bias and leads to more stable and reliable policy convergence.

2. **Dueling DQN**:
   Dueling DQN modifies the network architecture instead of the learning algorithm. It splits the final layers into two separate streams: one for the state Value function `V(s)` and another for the Advantage function `A(s, a)`. They are combined using: `Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))`. This helps the agent learn which states are inherently valuable regardless of the action taken (e.g., states far from the pit), which accelerates learning, particularly in environments like GridWorld where many actions do not significantly change the state's underlying value.

## HW3-3: Enhance DQN for Random Mode WITH Training Tips
**Implementation Strategy:**
For the fully random mode (all objects placed randomly), the problem space is vastly larger. We converted the PyTorch script into a structured **PyTorch Lightning** module (`LightningDQN`).

**Training Tips integrated:**
1. **Gradient Clipping**: Applied during `trainer.fit` (`gradient_clip_val=1.0`). This prevents the exploding gradient problem by capping the L2 norm of the gradients. It's crucial in RL since high-variance target Q-values can otherwise cause massive disruptive updates to network weights.
2. **Learning Rate Scheduling**: Implemented via `optim.lr_scheduler.StepLR` in `configure_optimizers`. This slowly reduces the learning rate over time, allowing the network to take larger steps initially and converge more delicately later, preventing the model from overshooting optimal weights as it approaches convergence.

## HW3-4: Rainbow DQN (Bonus)
**Analysis & Strategy for Random Mode GridWorld:**
The Random Mode presents a difficult challenge because the map changes entirely every episode. A single DQN improvement is usually insufficient. Rainbow DQN combines multiple independent improvements that complement each other.

**Implementation ("Rainbow Lite"):**
In our `hw3_4_rainbow_dqn.py` script, we constructed a streamlined Rainbow formulation:
- **Double DQN**: To mitigate overestimation.
- **Dueling Architecture**: To better evaluate states independent of actions in the highly variable random environment.
- **Multi-step (N-step) Returns**: Instead of calculating the target using just the immediate reward `r_t`, we calculate it using $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 \max Q$. This significantly speeds up the propagation of delayed rewards (like the +10 goal reward) to earlier states.

Combining these techniques enables robust learning even when the board configurations change entirely each game.
