import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from Gridworld import Gridworld

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
        return val + (adv - adv.mean(dim=1, keepdim=True))

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

def train_rainbow_lite(epochs=1500):
    print("Starting Training for Rainbow Lite (Double + Dueling + Multi-step) in Random Mode...")
    env = Gridworld(size=4, mode='random')
    policy_net = DuelingDQN()
    target_net = DuelingDQN()
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(5000)
    
    batch_size = 64
    gamma = 0.9
    n_step = 3 
    
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    update_target_every = 50
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    n_step_buffer = deque(maxlen=n_step)
    
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='random')
        state = env.board.render_np().reshape(64) + np.random.rand(64) / 10.0
        status = 1
        steps = 0
        n_step_buffer.clear()
        
        while status == 1 and steps < 50:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()
                    
            env.makeMove(action_set[action])
            reward = env.reward()
            done = True if reward in [10, -10] else False
            next_state = env.board.render_np().reshape(64) + np.random.rand(64) / 10.0
            
            n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(n_step_buffer) == n_step or done:
                n_reward = 0
                for i, (_, _, r, _, d) in enumerate(n_step_buffer):
                    n_reward += (gamma ** i) * r
                    if d: break
                
                n_s, n_a, _, _, _ = n_step_buffer[0]
                _, _, _, n_ns, n_d = n_step_buffer[-1]
                buffer.push(n_s, n_a, n_reward, n_ns, n_d)
                
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
                    next_action = policy_net(ns_batch).argmax(1).unsqueeze(1)
                    max_next_q = target_net(ns_batch).gather(1, next_action).squeeze()
                    expected_q = r_batch + (gamma ** n_step) * max_next_q * (1 - d_batch)
                    
                loss = loss_fn(q_value, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                
        if epoch % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (epoch+1) % 100 == 0:
            print(f"[Rainbow Lite] Epoch {epoch+1}/{epochs}, Epsilon: {epsilon:.4f}")

    torch.save(policy_net.state_dict(), "rainbow_dqn_random.pth")
    print("Rainbow Lite training finished and saved as rainbow_dqn_random.pth.")

if __name__ == '__main__':
    train_rainbow_lite()
