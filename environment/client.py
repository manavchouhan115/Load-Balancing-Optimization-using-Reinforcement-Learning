from dataclasses import dataclass
import numpy as np
from typing import Optional, List
from enum import Enum
import time

class RequestType(Enum):
    """Types of requests with different processing requirements"""
    LIGHT = 1 
    MEDIUM = 2
    HEAVY = 3

@dataclass
class Request:
    """Represents a single request from a client"""
    id: int
    arrival_time: float
    type: RequestType
    size: float 
    
    def __post_init__(self):
        self.processing_time = self.size * {
            RequestType.LIGHT: 1.0,
            RequestType.MEDIUM: 2.0,
            RequestType.HEAVY: 3.0
        }[self.type]

class Client:
    """
    Simulates client behavior by generating requests.
    Works with LoadBalancerEnv to simulate realistic traffic patterns.
    """
    def __init__(self, 
                 request_pattern: str = "uniform",
                 mean_request_rate: float = 1.0,
                 burst_probability: float = 0.2,
                 burst_rate_multiplier: float = 3.0):
        self.request_pattern = request_pattern
        self.mean_request_rate = mean_request_rate
        self.burst_probability = burst_probability
        self.burst_rate_multiplier = burst_rate_multiplier
        
        self.current_request_id = 0
        self.current_time = 0.0
        self.is_bursting = False
        
        self.type_probabilities = {
            RequestType.LIGHT: 0.6,
            RequestType.MEDIUM: 0.3,
            RequestType.HEAVY: 0.1
        }

    def _generate_request_size(self, request_type: RequestType) -> float:
        """Generate request size based on type"""
        base_sizes = {
            RequestType.LIGHT: (0.1, 0.3),
            RequestType.MEDIUM: (0.3, 0.7),
            RequestType.HEAVY: (0.7, 1.0)
        }
        min_size, max_size = base_sizes[request_type]
        return np.random.uniform(min_size, max_size)

    def _get_request_type(self) -> RequestType:
        """Randomly select request type based on probabilities"""
        types = list(RequestType)
        probabilities = [self.type_probabilities[t] for t in types]
        return np.random.choice(types, p=probabilities)

    def _get_current_rate(self) -> float:
        """Calculate current request rate based on pattern"""
        base_rate = self.mean_request_rate
        
        if self.request_pattern == "uniform":
            return base_rate
        
        elif self.request_pattern == "bursty":
            if not self.is_bursting and np.random.random() < self.burst_probability:
                self.is_bursting = True
            elif self.is_bursting and np.random.random() < 0.3:
                self.is_bursting = False
                
            return base_rate * self.burst_rate_multiplier if self.is_bursting else base_rate
        
        elif self.request_pattern == "periodic":
            phase = (self.current_time % 100) / 100
            rate_factor = 0.5 * (1 + np.sin(2 * np.pi * phase))
            return base_rate * (1 + rate_factor)
        
        return base_rate

    def generate_request(self, current_time: float) -> Optional[Request]:
        """Generate a new request based on current pattern and time"""
        self.current_time = current_time
        rate = self._get_current_rate()
        
        if np.random.random() < rate:
            self.current_request_id += 1
            request_type = self._get_request_type()
            
            return Request(
                id=self.current_request_id,
                arrival_time=current_time,
                type=request_type,
                size=self._generate_request_size(request_type)
            )
        
        return None

    def reset(self):
        """Reset client state"""
        self.current_request_id = 0
        self.current_time = 0.0
        self.is_bursting = False