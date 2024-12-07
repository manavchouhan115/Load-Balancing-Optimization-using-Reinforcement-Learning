from .base import BaseAgent
from environment.load_balancer import LoadBalancerState

class RoundRobinAgent(BaseAgent):
    def __init__(self, n_servers: int):
        super().__init__(n_servers)
        self.current_server = 0  # Keeps track of the next server in sequence

    def select_action(self, state: LoadBalancerState) -> int:
        # Select the current server and increment for the next request
        selected_server = self.current_server
        self.current_server = (self.current_server + 1) % self.n_servers
        return selected_server

    def update(self, state: LoadBalancerState, action: int, reward: float, 
               next_state: LoadBalancerState, done: bool):
        # No learning required for Round Robin, so we don't implement anything here
        pass
