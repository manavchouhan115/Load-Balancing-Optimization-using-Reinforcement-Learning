import numpy as np
from environment.load_balancer import LoadBalancerState
from .base import BaseAgent

class SARSAAgent(BaseAgent):
    def __init__(self, n_servers: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        super().__init__(n_servers)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def _state_to_key(self, state: LoadBalancerState) -> tuple:
        # Convert continuous state to a discrete representation for the Q-table
        return tuple(np.round(state.server_loads, 2))

    def select_action(self, state: LoadBalancerState) -> int:
        state_key = self._state_to_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_servers)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_servers)
        
        return np.argmax(self.q_table[state_key])

    def update(self, state: LoadBalancerState, action: int, 
               reward: float, next_state: LoadBalancerState, done: bool, next_action: int):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_servers)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_servers)
        
        # SARSA update rule
        current_q = self.q_table[state_key][action]
        next_q = self.q_table[next_state_key][next_action] if not done else 0
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.q_table[state_key][action] = new_q
