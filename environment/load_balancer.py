import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from .server import Server

@dataclass
class LoadBalancerState:
    server_loads: np.ndarray
    server_response_times: np.ndarray
    request_queue_length: int
    total_requests: int

class LoadBalancerEnv:
    def __init__(self, n_servers: int = 3, max_requests: int = 100):
        self.n_servers = n_servers
        self.max_requests = max_requests
        self.servers = [Server() for _ in range(n_servers)]
        self.current_step = 0
        self.reset()
    
    def reset(self) -> LoadBalancerState:
        self.current_step = 0
        for server in self.servers:
            server.reset()
        return self._get_state()
    
    def step(self, action: int) -> Tuple[LoadBalancerState, float, bool, float]:
        if action >= self.n_servers:
            raise ValueError(f"Invalid action {action}")
        
        response_time = self.servers[action].process_request()
        
        reward = self._calculate_reward(response_time)

        self.current_step += 1
        done = self.current_step >= self.max_requests
        
        return self._get_state(), reward, done, response_time
    
    def _get_state(self) -> LoadBalancerState:
        return LoadBalancerState(
            server_loads=np.array([s.current_load for s in self.servers]),
            server_response_times=np.array([s.avg_response_time for s in self.servers]),
            request_queue_length=sum(s.queue_length for s in self.servers),
            total_requests=self.current_step
        )
    
    def _calculate_reward(self, response_time: float) -> float:
        load_std = np.std([s.current_load for s in self.servers])
        return -(response_time + load_std)