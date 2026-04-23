# Robustness of Rational Learning Under Distorted Exposure

## Description
This repository contains the simulation code for the master's thesis
*Informational Lock-In in a Bandit Model of Biased Sampling*.
The project studies a two-source learning environment in which a rational learner updates beliefs correctly, but realized exposure is distorted before information is encountered.
The aim is to identify when distorted exposure preserves robust learning and when it produces
persistent ambiguity or convergence to an inferior source (informational lock-in).

## Research Question
Under what conditions is rational learning robust to distorted exposure,
and when does exposure distortion generate persistent ambiguity or informational lock-in?

## Repo Structure
- `src/`: core model components (bandit environment, algorithms, exposure mechanism, metrics, simulation)
- `experiments/`: runnable scripts for simulation grids, diagnostics, robustness checks, and plotting
- `results/data/`: generated CSV outputs from simulation runs
- `results/figures/`: generated PDF figures

## How To Reproduce The Figures
1. Activate environment and install dependencies (see section below).
2. Run the main Thompson Sampling grid:

```bash
python -m experiments.run_fragility_map
```

3. Create main thesis figures:

```bash
python -m experiments.plot_optimal_map
python -m experiments.plot_lockin_map
```

4. Run robustness grids and produce robustness figures:

```bash
python -m experiments.run_fragility_map_ucb1
python -m experiments.run_fragility_map_epsgreedy
python -m experiments.plot_ucb1_map
python -m experiments.plot_greedy_map
python -m experiments.plot_robustness_panel
```

## Environment
Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies are listed in `requirements.txt` (NumPy, Pandas, Matplotlib, Seaborn, tqdm, and related packages).

## Note
This project uses simulated data only.
All CSV files in `results/data/` are generated from model runs and can be reproduced using the scripts in `experiments/`.

## Output Files
Main outputs include:
- `results/data/fragility_map.csv`
- `results/figures/fragility_map_optimal.pdf`
- `results/figures/fragility_map_lockin.pdf`

Robustness outputs include:
- `results/data/fragility_map_ucb1.csv`
- `results/data/fragility_map_epsgreedy.csv`
- `results/figures/fragility_map_ucb1_optimal.pdf`
- `results/figures/fragility_map_ucb1_lockin.pdf`
- `results/figures/fragility_map_epsgreedy_optimal.pdf`
- `results/figures/fragility_map_epsgreedy_lockin.pdf`
- `results/figures/fragility_map_robustness_panel.pdf`

## Author 
Nadine Daum
Master's thesis: *Informational Lock-In in a Bandit Model of Biased Sampling*
