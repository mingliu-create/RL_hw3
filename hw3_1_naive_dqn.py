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

class NaiveDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(NaiveDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_naive_dqn():
    env = Gridworld(size=4, mode='static')
    model = NaiveDQN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(1000)
    
    epochs = 1000
    batch_size = 32
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    losses = []
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    print("Starting Training for Naive DQN (Static Mode)...")
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='static')
        state = env.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        status = 1
        steps = 0
        
        while status == 1 and steps < 50:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32))
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
                
                q_values = model(s_batch)
                q_value = q_values.gather(1, a_batch).squeeze()
                
                with torch.no_grad():
                    next_q_values = model(ns_batch)
                    max_next_q = torch.max(next_q_values, dim=1)[0]
                    expected_q = r_batch + gamma * max_next_q * (1 - d_batch)
                    
                loss = loss_fn(q_value, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses[-100:]):.4f}, Epsilon: {epsilon:.4f}")
            
    torch.save(model.state_dict(), "naive_dqn_static.pth")
    print("Training finished and model saved as naive_dqn_static.pth")

if __name__ == '__main__':
    train_naive_dqn()
