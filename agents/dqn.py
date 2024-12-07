import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from environment.load_balancer import LoadBalancerState
from .base import BaseAgent

# Neural Network for DQN
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseAgent):
    def __init__(self, n_servers: int, state_dim: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.1, memory_size: int = 10000, 
                 batch_size: int = 64):
        super().__init__(n_servers)
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Memory for experience replay
        self.memory = deque(maxlen=memory_size)

        # Neural networks
        self.q_network = QNetwork(state_dim, n_servers)
        self.target_network = QNetwork(state_dim, n_servers)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state: LoadBalancerState) -> int:
        """Select an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_servers)

        # Use the Q-network to select the action
        state_tensor = torch.FloatTensor(state.server_loads).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def train(self):
        """Train the Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute the current Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Compute the target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Update the target network to match the Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self, state: LoadBalancerState, action: int, reward: float, next_state: LoadBalancerState, done: bool):
        """Store the experience and train the network."""
        self.store_experience(state.server_loads, action, reward, next_state.server_loads, done)
        self.train()
        if done:
            self.update_target_network()
