import numpy as np
from collections import deque

class Server:
    def __init__(self, capacity: float = 1.0, mean_process_time: float = 0.1):
        self.capacity = capacity
        self.mean_process_time = mean_process_time
        self.current_load = 0.0
        self.request_queue = deque()
        self.response_times = []
    
    def process_request(self) -> float:
        process_time = np.random.exponential(self.mean_process_time)
        self.current_load = min(1.0, self.current_load + process_time)
        self.response_times.append(process_time)
        return process_time
    
    @property
    def avg_response_time(self) -> float:
        return np.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def queue_length(self) -> int:
        return len(self.request_queue)
    
    def reset(self):
        self.current_load = 0.0
        self.request_queue.clear()
        self.response_times.clear()