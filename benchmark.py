import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from environment.load_balancer import LoadBalancerEnv
from environment.client import Client

from agents.base import BaseAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.ppo import PPOAgent
from agents.round_robin import RoundRobinAgent


import warnings
warnings.filterwarnings("ignore")

class BenchmarkScenario:
    def __init__(self, 
                 name: str,
                 client_config: Dict[str, Any],
                 duration: int = 1000):
        self.name = name
        self.client_config = client_config
        self.duration = duration

class Benchmark:
    def __init__(self, 
                 algorithms: List[str],
                 n_servers: int = 3,
                 n_episodes: int = 1000,
                 n_runs: int = 5,
                 save_dir: str = "benchmark_results"):
        self.algorithms = algorithms
        self.n_servers = n_servers
        self.n_episodes = n_episodes
        self.n_runs = n_runs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.scenarios = [
            BenchmarkScenario(
                name="uniform_light_load",
                client_config={
                    "request_pattern": "uniform",
                    "mean_request_rate": 0.5,
                }
            ),
            BenchmarkScenario(
                name="uniform_heavy_load",
                client_config={
                    "request_pattern": "uniform",
                    "mean_request_rate": 2.0,
                }
            ),
            BenchmarkScenario(
                name="bursty_traffic",
                client_config={
                    "request_pattern": "bursty",
                    "mean_request_rate": 1.0,
                    "burst_probability": 0.2,
                    "burst_rate_multiplier": 3.0,
                }
            ),
            BenchmarkScenario(
                name="periodic_traffic",
                client_config={
                    "request_pattern": "periodic",
                    "mean_request_rate": 1.0,
                }
            )
        ]

        self.metrics = [
            "average_response_time",
            "load_balance_std",
            "total_reward"
        ]

    def get_agent(self, algorithm: str) -> BaseAgent:
        """Initialize agent based on algorithm name."""
        if algorithm == "q_learning":
            return QLearningAgent(n_servers=self.n_servers)
        elif algorithm == "sarsa":
            return SARSAAgent(n_servers=self.n_servers)
        elif algorithm == "ppo":
            return PPOAgent(n_servers=self.n_servers)
        elif algorithm == "round_robin":
            return RoundRobinAgent(n_servers=self.n_servers)

        raise ValueError(f"Unknown algorithm: {algorithm}")

    def run_single_evaluation(self, 
                            agent: BaseAgent, 
                            env: LoadBalancerEnv,
                            client: Client,
                            duration: int) -> Dict[str, float]:
        """Run a single evaluation episode."""
        state = env.reset()
        client.reset()
        
        metrics = defaultdict(list)
        request_times = defaultdict(list)
        
        for t in range(duration):
            request = client.generate_request(float(t))
            
            if request:
                action = agent.select_action(state)
                next_state, reward, done, response_time = env.step(action)
                
                metrics["rewards"].append(reward)
                metrics["load_balance"].append(np.std(state.server_loads))
                
                request_times[request.type].append(response_time)
                
                state = next_state
        
        results = {
            "average_response_time": np.mean([t for times in request_times.values() for t in times]),
            "load_balance_std": np.mean(metrics["load_balance"]),
            "total_reward": sum(metrics["rewards"]),
        }
                
        return results

    def run_benchmark(self) -> pd.DataFrame:
        """Run full benchmark suite."""
        results = []
        
        for algorithm in self.algorithms:
            for scenario in self.scenarios:
                print(f"\nBenchmarking {algorithm} on {scenario.name}")
                
                for run in tqdm(range(self.n_runs)):

                    env = LoadBalancerEnv(n_servers=self.n_servers)
                    client = Client(**scenario.client_config)
                    agent = self.get_agent(algorithm)

                    for episode in range(self.n_episodes):
                        state = env.reset()
                        client.reset()
                        done = False
                        t = 0
                        
                        while not done and t < scenario.duration:
                            request = client.generate_request(float(t))
                            
                            if request:
                                action = agent.select_action(state)
                                next_state, reward, done, _ = env.step(action)
                                if algorithm == "q_learning":
                                    agent.update(state, action, reward, next_state, done)
                                elif algorithm == "sarsa":
                                    next_action = agent.select_action(next_state)
                                    agent.update(state, action, reward, next_state, done, next_action)
                                elif algorithm == "ppo":
                                    state_value = agent.update_value_estimate(state)
                                    agent.store_transition(reward, state_value)
                                state = next_state
                            
                            t += 1
                        # For PPO, update the policy after the episode ends
                        if algorithm == "ppo":
                            agent.end_episode()
                    # Evaluation phase
                    eval_metrics = self.run_single_evaluation(
                        agent, env, client, scenario.duration
                    )
                    
                    results.append({
                        "algorithm": algorithm,
                        "scenario": scenario.name,
                        "run": run,
                        **eval_metrics
                    })
        
        return pd.DataFrame(results)
    
    def generate_plots(self, results: pd.DataFrame):
        """Generate visualization plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = self.save_dir / "plots" / timestamp
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-paper')
        
        for metric in self.metrics:
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=results,
                x="scenario",
                y=metric,
                hue="algorithm",
                ci="sd"
            )
            if metric == "average_response_time":
                plt.yscale('log')
            plt.title(f"{metric.replace('_', ' ').title()} by Algorithm and Scenario")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{metric}_comparison.png")
            plt.close()
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        pivot_data = results.pivot_table(
            values="load_balance_std",
            index="scenario",
            columns="algorithm",
            aggfunc="mean"
        )
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd"
        )
        plt.title("Load Balance Standard Deviation")
        plt.tight_layout()
        plt.savefig(plots_dir / "load_balance_heatmap.png")
        plt.close()

    def save_results(self, results: pd.DataFrame):
        """Save benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results.to_csv(self.save_dir / f"benchmark_results_{timestamp}.csv", index=False)
        
        summary = results.groupby(["algorithm", "scenario"]).agg({
            metric: ["mean", "std", "min", "max"] for metric in self.metrics
        }).round(4)
        
        summary.to_csv(self.save_dir / f"benchmark_summary_{timestamp}.csv")
        
        config = {
            "n_servers": self.n_servers,
            "n_episodes": self.n_episodes,
            "n_runs": self.n_runs,
            "scenarios": [
                {
                    "name": scenario.name,
                    "config": scenario.client_config,
                    "duration": scenario.duration
                }
                for scenario in self.scenarios
            ],
            "algorithms": self.algorithms
        }
        
        with open(self.save_dir / f"benchmark_config_{timestamp}.json", "w") as f:
            json.dump(config, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Load Balancer Benchmark Suite')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['q_learning'],
                       help='Algorithms to benchmark')
    parser.add_argument('--n-servers', type=int, default=3,
                       help='Number of servers')
    parser.add_argument('--n-episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--n-runs', type=int, default=5,
                       help='Number of runs per configuration')
    parser.add_argument('--save-dir', type=str, default='benchmark_results',
                       help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    benchmark = Benchmark(
        algorithms=args.algorithms,
        n_servers=args.n_servers,
        n_episodes=args.n_episodes,
        n_runs=args.n_runs,
        save_dir=args.save_dir
    )

    results = benchmark.run_benchmark()
    
    benchmark.generate_plots(results)

    benchmark.save_results(results)
    
    print(f"\nBenchmark complete. Results saved in {args.save_dir}")

if __name__ == "__main__":
    main()