# OpenMotor-Optimizer

**Physics-guided optimization of solid rocket motors with OpenMotor integration and GPU-accelerated search.**

OpenMotor-Optimizer is a structured experimentation platform for solid rocket motor design optimization. Drawing inspiration from modern machine learning AutoML pipelines, this repository transforms the traditional `generate → simulate → print` workflow into a rigorous engineering exploration process:

`constraints → geometry generator → simulation → evaluation → design search`

## Features

- **Decoupled Physics Engine**: Interacts cleanly with OpenMotor through adapter layers, without reimplementing core physics equations.
- **GPU-Accelerated Evaluation**: Leverages Numba and PyTorch for parallel mass-evaluation of geometry configurations.
- **Persistent Experiment Tracking**: Organizes optimization configurations, run records, thrust curves, Pareto frontiers, and exported `.ric` files via an experiment orchestrator.
- **Structured Optimizers**: Implements Monte-Carlo, Genetic Algorithms, and Bayesian Search for solid rocket grains.
- **UI Dashboard**: A built-in Streamlit dashboard mimicking modern ML tracking tools to visualize optimal candidates and performance metrics.

## Architecture

The environment relies on isolating candidate geometry generation from physics simulation and parallel orchestration:

```
CPU
  ├─ Geometry parameter generation
  ├─ Experiment orchestration
  └─ OpenMotor precise sequential physics control

GPU
  ├─ Batch candidate heuristic evaluation
  ├─ Optimization space sampling
  └─ Parallel constraint filtering
```

## Installation

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/Digvijay05/OpenMotor-Optimizer.git
cd OpenMotor-Optimizer
pip install -r requirements.txt
pip install -e .
```

*Note: You must have an established OpenMotor module in your Python path to execute true physical simulations.*

## Running Experiments

Experiments are configured via intuitive YAML files located in `experiments/configs/`.

Execute an experiment:
```bash
python run_experiment.py experiments/configs/bates_high_thrust.yaml
```

This will run the designated optimizer on the defined constraints, producing graphical outputs, structural metadata CSVs, and optimal `.ric` configurations in `experiments/results/`.

## Dashboard Application

To launch the analysis UI:

```bash
streamlit run openmotor_optimizer/ui/streamlit_app.py
```
