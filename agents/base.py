from abc import ABC, abstractmethod
from environment.load_balancer import LoadBalancerState

class BaseAgent(ABC):
    def __init__(self, n_servers: int):
        self.n_servers = n_servers
    
    @abstractmethod
    def select_action(self, state: LoadBalancerState) -> int:
        pass
    
    @abstractmethod
    def update(self, state: LoadBalancerState, action: int, 
               reward: float, next_state: LoadBalancerState, done: bool):
        pass