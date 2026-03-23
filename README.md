# How Algorithms Get Stuck
### or why small recommendation distortions can derail our rational learning

This repo contains the code for my thesis: **Robustness of Rational Learning Under Distorted Exposure**: Informational Lock-In in a Bandit Model of Biased Sampling (Author: Nadine Daum)

## Quick Start
Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import numpy, pandas, matplotlib; print('env OK')"
```

## Reproduce Results
1. Run simulation grid (creates CSV):

```bash
python -m experiments.run_fragility_map
```

2. Create thesis figures (PDF only):

```bash
python -m experiments.plot_optimal_map
python -m experiments.plot_lockin_map
```

Outputs:
- `results/data/fragility_map.csv`
- `results/figures/fragility_map_optimal.pdf`
- `results/figures/fragility_map_lockin.pdf`

## Project Structure
- `src/`: core model logic (bandit, learner, exposure, metrics, simulation)
- `experiments/`: runnable scripts for experiments and plots
- `results/data/`: generated CSV files
- `results/figures/`: generated PDF figures

Optional diagnostics: `python -m experiments.run_batch_test`, `python -m experiments.run_diagnostic_test`, `python -m experiments.run_minigrid_test`.
