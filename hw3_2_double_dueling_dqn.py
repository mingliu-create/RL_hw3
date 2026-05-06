import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from Gridworld import Gridworld

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class DoubleDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DoubleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        
        self.val1 = nn.Linear(150, 100)
        self.val2 = nn.Linear(100, 1)
        
        self.adv1 = nn.Linear(150, 100)
        self.adv2 = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        val = self.val2(torch.relu(self.val1(x)))
        adv = self.adv2(torch.relu(self.adv1(x)))
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return val + (adv - adv.mean(dim=1, keepdim=True))

def train_model(model_type='double', epochs=1000):
    print(f"\nStarting Training for {model_type.upper()} DQN (Player Mode)...")
    env = Gridworld(size=4, mode='player')
    
    if model_type == 'double':
        policy_net = DoubleDQN()
        target_net = DoubleDQN()
    else:
        policy_net = DuelingDQN()
        target_net = DuelingDQN()
        
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(2000)
    
    batch_size = 32
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    update_target_every = 50
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='player')
        state = env.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        status = 1
        steps = 0
        
        while status == 1 and steps < 50:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()
                    
            env.makeMove(action_set[action])
            reward = env.reward()
            done = True if reward in [10, -10] else False
            
            next_state = env.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            buffer.push(state.squeeze(), action, reward, next_state.squeeze(), done)
            state = next_state
            steps += 1
            
            if done:
                status = 0
                
            if len(buffer) > batch_size:
                s_batch, a_batch, r_batch, ns_batch, d_batch = buffer.sample(batch_size)
                s_batch = torch.tensor(s_batch, dtype=torch.float32)
                a_batch = torch.tensor(a_batch, dtype=torch.int64).unsqueeze(1)
                r_batch = torch.tensor(r_batch, dtype=torch.float32)
                ns_batch = torch.tensor(ns_batch, dtype=torch.float32)
                d_batch = torch.tensor(d_batch, dtype=torch.float32)
                
                q_value = policy_net(s_batch).gather(1, a_batch).squeeze()
                
                with torch.no_grad():
                    if model_type == 'double':
                        next_action = policy_net(ns_batch).argmax(1).unsqueeze(1)
                        max_next_q = target_net(ns_batch).gather(1, next_action).squeeze()
                    else:
                        max_next_q = target_net(ns_batch).max(1)[0]
                        
                    expected_q = r_batch + gamma * max_next_q * (1 - d_batch)
                    
                loss = loss_fn(q_value, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if epoch % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (epoch+1) % 100 == 0:
            print(f"[{model_type.upper()}] Epoch {epoch+1}/{epochs}, Epsilon: {epsilon:.4f}")

    torch.save(policy_net.state_dict(), f"{model_type}_dqn_player.pth")
    print(f"{model_type.upper()} training finished and saved as {model_type}_dqn_player.pth.")

if __name__ == '__main__':
    train_model('double', 1500)
    train_model('dueling', 1500)
