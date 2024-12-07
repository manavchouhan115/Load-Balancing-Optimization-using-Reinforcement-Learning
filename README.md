# AI Planning Project

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MadhavWalia/balanceRL.git
cd balanceRL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the benchmarking script using the following command structure:
```bash
python benchmark.py --algorithms <List of RL Algorithms> --n-episodes <Number of Training Episodes> --n-runs <Runs Per Config>
```

### Parameters

- `--algorithms`: List of reinforcement learning algorithms to benchmark
- `--n-episodes`: Number of training episodes per run
- `--n-runs`: Number of runs per configuration (for statistical significance)

### Example

To run a benchmark of Q-Learning for 1000 episodes, with 5 runs:
```bash
python benchmark.py --algorithms q_learning --n-episodes 1000 --n-runs 5
```

### Multiple Algorithms

You can benchmark multiple algorithms by listing them after the `--algorithms` flag:
```bash
python benchmark.py --algorithms q_learning sarsa dqn --n-episodes 1000 --n-runs 5
```
