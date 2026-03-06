<![CDATA[# 🚀 OpenMotor-Optimizer

**Physics-guided optimization of solid rocket motors with OpenMotor integration and GPU-accelerated search.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenMotor](https://img.shields.io/badge/OpenMotor-Compatible-orange.svg)](https://github.com/reilleya/openMotor)

---

OpenMotor-Optimizer is a structured experimentation platform for **solid rocket motor design optimization**. Inspired by modern machine-learning AutoML pipelines, it transforms the traditional _generate → simulate → print_ workflow into a rigorous, reproducible engineering exploration process:

```
constraints → physics-guided generator → GPU filtering → OpenMotor simulation → evaluation → design export
```

> **Note:** This tool is a _wrapper_ around the open-source [OpenMotor](https://github.com/reilleya/openMotor) simulation library — it does **not** reimplement any core physics. All ballistic simulations are performed by OpenMotor itself.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Dashboard](#dashboard)
- [Supported Grain Geometries](#supported-grain-geometries)
- [Optimizers](#optimizers)
- [Performance Metrics](#performance-metrics)
- [Export Format](#export-format)
- [GPU Acceleration](#gpu-acceleration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Decoupled Physics Engine** | Interacts cleanly with OpenMotor through adapter layers, without reimplementing core physics equations. |
| **GPU-Accelerated Pre-Filtering** | Leverages PyTorch tensors to batch-evaluate millions of candidate geometries on GPU before sending survivors to physical simulation. |
| **Physics-Guided Search** | Uses Kn (Klemmung) rules-of-thumb and mass conservation to derive intelligent search bounds rather than naïve random sampling. |
| **3 Optimization Strategies** | Monte Carlo (GPU), Genetic Algorithm, and Bayesian (Optuna) optimizers, configurable per experiment. |
| **Multiple Grain Geometries** | Supports BATES, Finocyl, and Star grain profiles with full parameterization. |
| **Persistent Experiment Tracking** | Organizes configs, run records, thrust curves, Pareto frontiers, and exported `.ric` files per experiment. |
| **Streamlit Dashboard** | Built-in UI for experiment visualization: Pareto plots, KPI cards, data tables, and one-click `.ric` download. |
| **`.ric` Export** | Directly exports optimized motor designs in OpenMotor-compatible `.ric` format for immediate visualization. |

---

## Architecture

The platform separates _cheap vectorized filtering_ (GPU) from _expensive physics simulation_ (CPU/OpenMotor):

```
┌──────────────────────────────────────────────────────────┐
│  YAML Config                                             │
│    └─ constraints, propellant, optimizer, objective       │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  Physics-Guided Bounds  (CPU)                            │
│    └─ Kn rules, mass conservation → smart search space   │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  Candidate Generation  (CPU)                             │
│    └─ Random / Gaussian sampling over bounded space      │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  GPU Batch Filter  (GPU / PyTorch)                       │
│    └─ Vectorized geometric constraint checks             │
│    └─ Rejects invalid core-to-OD ratios, fin reach, etc. │
└──────────┬───────────────────────────────────────────────┘
           │  survivors only
           ▼
┌──────────────────────────────────────────────────────────┐
│  OpenMotor Simulation  (CPU, multiprocessing)            │
│    └─ Assembles Motor → runs internal ballistics sim     │
│    └─ Extracts: thrust, impulse, Isp, burn time, etc.    │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  Outputs                                                 │
│    ├─ motors.csv        (all valid motor profiles)       │
│    ├─ best_motor.ric    (OpenMotor-compatible export)    │
│    └─ Dashboard         (Streamlit visualization)        │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
OpenMotor-Optimizer/
├── openmotor_optimizer/            # Main Python package
│   ├── compute/                    # GPU scheduling & device management
│   │   └── gpu_scheduler.py
│   ├── evaluation/                 # Performance metrics dataclass
│   │   └── performance_metrics.py
│   ├── exporters/                  # .ric file export to OpenMotor format
│   │   └── ric_exporter.py
│   ├── generator/                  # Candidate parameter generation
│   │   ├── physics_guided.py       # Kn-based bound calculation
│   │   └── random_generator.py     # Random/Gaussian vector sampling
│   ├── grains/                     # Grain geometry definitions
│   │   ├── bates.py
│   │   ├── finocyl.py
│   │   └── star.py
│   ├── optimizer/                  # Optimization engines
│   │   ├── montecarlo.py           # GPU-accelerated Monte Carlo
│   │   ├── genetic.py              # Genetic algorithm
│   │   └── bayesian.py             # Bayesian optimization (Optuna)
│   ├── propellants/                # OpenMotor library adapter
│   │   └── openmotor_adapter.py
│   ├── simulation/                 # Simulation runners
│   │   ├── openmotor_runner.py     # Sequential OpenMotor sim wrapper
│   │   └── gpu_batch_runner.py     # PyTorch batch filtering
│   └── ui/                         # Streamlit dashboard
│       └── streamlit_app.py
├── experiments/
│   ├── configs/                    # YAML experiment configurations
│   │   └── high_thrust_kndx.yaml   # Example: KNDX high-thrust search
│   └── results/                    # Auto-generated per-run outputs
├── run_experiment.py               # CLI entry point
├── test_single.py                  # Single-motor debug test
├── requirements.txt
├── pyproject.toml
├── setup.py
├── LICENSE                         # MIT
└── README.md
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or later |
| **OpenMotor** | Must be importable as `motorlib`. See [Installation](#installation). |
| **CUDA** _(optional)_ | NVIDIA GPU + CUDA toolkit for GPU-accelerated filtering. Falls back to CPU if unavailable. |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Digvijay05/OpenMotor-Optimizer.git
cd OpenMotor-Optimizer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs all core dependencies **including** OpenMotor directly from GitHub:

| Package | Purpose |
|---------|---------|
| `numpy` ≥ 1.26 | Array operations |
| `scipy` ≥ 1.11 | Savitzky–Golay filtering, optimization |
| `pandas` ≥ 2.1 | Results DataFrames |
| `matplotlib` ≥ 3.8 | Plotting |
| `numba` ≥ 0.58 | JIT-compiled numeric kernels |
| `torch` ≥ 2.0 | GPU tensor operations for batch filtering |
| `streamlit` ≥ 1.28 | Interactive dashboard |
| `pyyaml` ≥ 6.0 | YAML config parsing |
| `optuna` ≥ 3.4 | Bayesian hyperparameter search |
| `openMotor` | Solid rocket motor simulation engine (installed from GitHub) |

### 4. Install the package in editable mode

```bash
pip install -e .
```

### 5. Verify OpenMotor is accessible

```bash
python -c "from motorlib.motor import Motor; print('OpenMotor OK')"
```

If this fails, set the `OPENMOTOR_PATH` environment variable to the root of your local OpenMotor checkout:

```bash
# Windows
set OPENMOTOR_PATH=C:\path\to\openMotor

# macOS / Linux
export OPENMOTOR_PATH=/path/to/openMotor
```

---

## Configuration

Experiments are configured via YAML files in `experiments/configs/`. Here is the included example:

### `experiments/configs/high_thrust_kndx.yaml`

```yaml
experiment_name: high_thrust_kndx

constraints:
  target_mass_kg: 0.1
  max_pressure_Pa: 4826330     # 700 psi
  max_mass_flux: 500           # kg/(m²·s)
  grain_od_m: 0.035            # 35 mm outer diameter
  total_grains: 4
  mode: fast                   # "fast" = BATES only, "full" = BATES + Finocyl

propellant:
  name: "KNDX"
  density: 1879.0              # kg/m³
  tabs:
    - minPressure: 100000      # Pa
      maxPressure: 10300000    # Pa
      a: 0.000001713           # burn rate coefficient
      n: 0.619                 # burn rate exponent
      k: 1.1308                # ratio of specific heats
      t: 1710.0                # combustion temperature (K)
      m: 42.39                 # molar mass (g/mol)

optimizer:
  type: montecarlo             # montecarlo | genetic | bayesian
  samples: 5000                # total candidates to generate
  gpu_batch_size: 1024

objective:
  maximize: peak_thrust
```

### Key Configuration Fields

| Field | Description | Example |
|-------|-------------|---------|
| `constraints.mode` | `"fast"` searches BATES-only space (4 dims); `"full"` adds Finocyl parameters (9 dims). | `fast` |
| `constraints.grain_od_m` | Motor casing outer diameter in meters. | `0.035` |
| `constraints.total_grains` | Number of grain segments stacked in the motor. | `4` |
| `optimizer.type` | Optimization strategy: `montecarlo`, `genetic`, or `bayesian`. | `montecarlo` |
| `optimizer.samples` | Number of candidate geometries to generate. | `5000` |
| `propellant.tabs` | Burn-rate tables following [St. Robert's law](https://en.wikipedia.org/wiki/Solid-propellant_rocket#Burn_rate). | See above |

---

## Running Experiments

### Basic usage

```bash
python run_experiment.py experiments/configs/high_thrust_kndx.yaml
```

### What happens

1. **Config is loaded** — constraints, propellant, optimizer settings parsed from YAML.
2. **Physics-guided bounds** — `calculate_initial_bounds()` derives sensible geometric ranges using Kn rules.
3. **Candidate generation** — Random vectors sampled within the computed bounds on CPU.
4. **GPU filtering** — PyTorch applies vectorized geometric constraint checks, rejecting infeasible designs instantly.
5. **OpenMotor simulation** — Surviving candidates are simulated in parallel using `multiprocessing.Pool`. Each worker assembles a full `Motor` object and runs the internal ballistics solver.
6. **Results export** — Valid motors are saved to `experiments/results/<name>_<timestamp>/`:
   - `motors.csv` — all valid motor profiles with performance metrics.
   - `best_motor.ric` — the highest-performing design in OpenMotor-compatible format.

### Example output

```
==================================================
Starting Experiment: high_thrust_kndx
Results Directory: experiments/results/high_thrust_kndx_20260307_001500
==================================================

Starting GPU-Accelerated Monte Carlo Search: 5000 samples.
GPU Filtering complete. 3847/5000 passed geometric checks (0.12s).
Spawning 8 physical workers for OpenMotor simulation...
  Processed 100/3847...
  Processed 200/3847...
  ...
Simulation completed. 1423 valid motor profiles generated.
Saved 1423 simulated motors to experiments/results/high_thrust_kndx_20260307_001500/motors.csv

Extracting Best Design...
Saved Best Profile to experiments/results/high_thrust_kndx_20260307_001500/best_motor.ric

Experiment Complete.
```

---

## Dashboard

Launch the interactive Streamlit dashboard to browse experiment results:

```bash
streamlit run openmotor_optimizer/ui/streamlit_app.py
```

### Dashboard features

- **Experiment selector** — browse all completed runs from the sidebar
- **KPI cards** — max peak thrust, max total impulse, max burn time, average Isp
- **Pareto frontier plot** — total impulse vs. peak thrust, color-coded by chamber pressure
- **Data table** — top 50 motor profiles sorted by objective
- **`.ric` download** — one-click download of the best motor for loading into OpenMotor

The dashboard automatically scans `experiments/results/` for completed runs.

---

## Supported Grain Geometries

| Geometry | Module | Parameters |
|----------|--------|------------|
| **BATES** | `grains/bates.py` | Outer diameter, length, core diameter, inhibited ends |
| **Finocyl** | `grains/finocyl.py` | Outer diameter, length, core diameter, fin count, fin width, fin length |
| **Star** | `grains/star.py` | Outer diameter, length, core diameter, point count, point width, point length |

In `mode: fast`, only BATES geometries are explored (4-dimensional search).
In `mode: full`, BATES + Finocyl geometries are explored (9-dimensional search).

---

## Optimizers

### Monte Carlo (`montecarlo`)

The default optimizer. Generates a massive random candidate matrix, filters it on GPU, then simulates survivors in parallel:

- **Strengths:** Simple, embarrassingly parallel, excellent for initial exploration.
- **Best for:** First-pass searches, understanding the design space.

### Genetic Algorithm (`genetic`)

Evolutionary search with crossover, mutation, and selection:

- **Strengths:** Converges toward optimal regions, handles multi-objective.
- **Best for:** Refining candidates found by Monte Carlo.

### Bayesian Optimization (`bayesian`)

Uses [Optuna](https://optuna.readthedocs.io/) with Tree-structured Parzen Estimator (TPE):

- **Strengths:** Sample-efficient, models the objective surface.
- **Best for:** Expensive evaluations where simulation budget is limited.

---

## Performance Metrics

Every valid simulation extracts the following metrics:

| Metric | Unit | Description |
|--------|------|-------------|
| `peak_thrust_N` | N | Maximum instantaneous thrust |
| `average_thrust_N` | N | Mean thrust over burn |
| `burn_time_s` | s | Total burn duration |
| `total_impulse_Ns` | N·s | Integral of thrust over time |
| `peak_pressure_Pa` | Pa | Maximum chamber pressure |
| `peak_mass_flux` | kg/(m²·s) | Maximum mass flux through port |
| `specific_impulse_s` | s | Total impulse / (propellant mass × g₀) |
| `propellant_mass_kg` | kg | Initial propellant mass |

These are defined in `openmotor_optimizer/evaluation/performance_metrics.py` as a `@dataclass`.

---

## Export Format

Optimized designs are exported as `.ric` files — the native JSON-based format used by OpenMotor.

A generated `.ric` contains:
- Format version
- Propellant definition (burn-rate tabs, density)
- Nozzle geometry (throat/exit diameters, convergent/divergent angles, efficiency)
- Grain stack (type-specific properties per grain segment)

To open an exported `.ric`:
1. Launch OpenMotor
2. **File → Open** → select the `.ric` file
3. The motor configuration loads with all grain segments, nozzle, and propellant data

---

## GPU Acceleration

GPU acceleration is used for **pre-filtering only** — the actual physics simulation still runs sequentially on CPU via OpenMotor.

### How it works

1. Candidate parameter vectors are moved to a PyTorch tensor on the GPU.
2. Vectorized constraint checks (core < OD, fin reach within casing, etc.) run in parallel.
3. Only geometrically valid candidates are transferred back to CPU for simulation.

### Auto-detection

The system automatically detects available hardware:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

If no CUDA GPU is available, filtering runs on CPU — still using PyTorch's vectorized operations for speed.

### Checking your GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Troubleshooting

### `Could not import motorlib`

OpenMotor is not on your Python path. Either:
- Install it: `pip install git+https://github.com/reilleya/openMotor.git`
- Or set `OPENMOTOR_PATH` to your local clone

### `BrokenProcessPool` during Monte Carlo

A specific motor configuration is causing a segfault in the C++ layer of OpenMotor. The GPU pre-filter catches most of these, but edge cases can slip through. Reduce `samples` or switch to `genetic`/`bayesian` optimizer.

### GPU not detected

Verify your PyTorch installation supports CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, install the CUDA-enabled version: https://pytorch.org/get-started/locally/

### `.ric` file won't open in OpenMotor

Ensure `formatVersion` in the exported file matches your OpenMotor version. The exporter defaults to version `4`.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Built for the rocketry community — from constraints to flight-ready grain designs.</em>
</p>
]]>
