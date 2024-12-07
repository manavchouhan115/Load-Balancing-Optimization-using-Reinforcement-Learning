import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.load_balancer import LoadBalancerState
from .base import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, n_servers: int, learning_rate: float = 0.001, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, update_epochs: int = 10):
        super().__init__(n_servers)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

        # Neural networks
        self.policy_net = self.build_network(n_servers)
        self.value_net = self.build_network(1)  # Output a single value for V(s)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Data storage for trajectory
        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []

    def build_network(self, output_size: int):
        """Builds a simple neural network with 2 hidden layers"""
        return nn.Sequential(
            nn.Linear(self.n_servers, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def select_action(self, state: LoadBalancerState) -> int:
        state_tensor = torch.tensor(state.server_loads, dtype=torch.float32)
        logits = self.policy_net(state_tensor)
        action_distribution = torch.distributions.Categorical(logits=logits)
        
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        
        # Store data for PPO update
        self.states.append(state_tensor)
        self.actions.append(action)
        self.log_probs.append(log_prob)

        return action.item()

    def update(self):
        # Calculate discounted rewards-to-go and advantages
        rewards_to_go, advantages = self.compute_advantages()

        # Loop over the number of update epochs
        for _ in range(self.update_epochs):
            # Loop over each data point in the collected trajectories
            for state, action, old_log_prob, reward_to_go, advantage in zip(
                    self.states, self.actions, self.log_probs, rewards_to_go, advantages):
            
                # Calculate new log_prob for the selected action
                logits = self.policy_net(state)
                action_distribution = torch.distributions.Categorical(logits=logits)
                log_prob = action_distribution.log_prob(action)
            
                # Calculate the ratio for PPO clipping
                ratio = torch.exp(log_prob - old_log_prob.detach())  # Detach old_log_prob to prevent in-place modification
            
                # Clipping the ratio to stay within the range
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                # Ensure advantage is detached if needed to prevent in-place operations
                unclipped_objective = ratio * advantage.detach()
                clipped_objective = clipped_ratio * advantage.detach()
                policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()

                # Compute policy loss and backpropagate
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Compute the value loss and update the value network
                value_loss = nn.MSELoss()(self.value_net(state), reward_to_go)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        # Clear the memory to prepare for the next episode
        self.clear_memory()


    def compute_advantages(self):
        """Compute rewards-to-go and advantages."""
        rewards_to_go = []
        advantages = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            rewards_to_go.insert(0, G)

        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32)
        values = torch.cat(self.values).detach()
        advantages = rewards_to_go - values  # Advantage = rewards-to-go - baseline value

        return rewards_to_go, advantages

    def store_transition(self, reward, state_value):
        """Store the reward and the state value during the episode."""
        self.rewards.append(reward)
        self.values.append(state_value)

    def clear_memory(self):
        """Clear all stored values."""
        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []

    def end_episode(self):
        """Trigger the end of an episode, which is when the PPO updates are performed."""
        self.update()

    def update_value_estimate(self, state: LoadBalancerState):
        """Get the value estimate for the current state."""
        state_tensor = torch.tensor(state.server_loads, dtype=torch.float32)
        value = self.value_net(state_tensor)
        return value
