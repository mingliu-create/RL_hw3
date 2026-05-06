import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import random
from collections import deque
from Gridworld import Gridworld

class RLDataset(Dataset):
    def __init__(self, buffer, sample_size):
        self.buffer = buffer
        self.sample_size = sample_size
        
    def __len__(self):
        return self.sample_size
        
    def __getitem__(self, idx):
        batch = random.sample(self.buffer, 1)[0]
        state, action, reward, next_state, done = batch
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.int64),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

class LightningDQN(pl.LightningModule):
    def __init__(self, input_dim=64, output_dim=4):
        super(LightningDQN, self).__init__()
        self.save_hyperparameters()
        
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.buffer = deque(maxlen=5000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
        self.env = Gridworld(size=4, mode='random')
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.loss_fn = nn.MSELoss()
        
        print("Populating initial replay buffer...")
        state = self.env.board.render_np().reshape(64) + np.random.rand(64) / 10.0
        for _ in range(500):
            action = random.randint(0, 3)
            self.env.makeMove(self.action_set[action])
            reward = self.env.reward()
            done = True if reward in [10, -10] else False
            next_state = self.env.board.render_np().reshape(64) + np.random.rand(64) / 10.0
            self.buffer.append((state, action, reward, next_state, done))
            if done:
                self.env = Gridworld(size=4, mode='random')
                state = self.env.board.render_np().reshape(64) + np.random.rand(64) / 10.0
            else:
                state = next_state

    def forward(self, x):
        return self.policy_net(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        # Training Tip: Learning Rate Scheduling
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, 200) 
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=0)

    def training_step(self, batch, batch_idx):
        s, a, r, ns, d = batch
        
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                state_t = torch.tensor(self.env.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0, dtype=torch.float32, device=self.device)
                q_values = self(state_t)
                action = torch.argmax(q_values).item()
                
        curr_state = self.env.board.render_np().reshape(64) + np.random.rand(64) / 10.0
        self.env.makeMove(self.action_set[action])
        reward = self.env.reward()
        done = True if reward in [10, -10] else False
        next_state = self.env.board.render_np().reshape(64) + np.random.rand(64) / 10.0
        
        self.buffer.append((curr_state, action, reward, next_state, done))
        if done:
            self.env = Gridworld(size=4, mode='random')
            
        q_value = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0]
            expected_q = r + self.gamma * max_next_q * (1 - d)
            
        loss = self.loss_fn(q_value, expected_q)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.current_epoch % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == '__main__':
    model = LightningDQN()
    # Training Tip: Gradient Clipping (gradient_clip_val)
    trainer = pl.Trainer(max_epochs=50, gradient_clip_val=1.0)
    print("Starting Training for PyTorch Lightning DQN (Random Mode)...")
    trainer.fit(model)
    trainer.save_checkpoint("lightning_dqn_random.ckpt")
    print("Training finished and saved as lightning_dqn_random.ckpt.")
