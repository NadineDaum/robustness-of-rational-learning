# Robustness of Rational Learning Under Distorted Exposure
**Author: Nadine Daum** 

This repo contains the simulation code for my master’s thesis: **Robustness of Rational Learning Under Distorted Exposure: Informational Lock-In in a Two-Source Bandit Model**.

The project studies a stylized two-source learning environment in which a rational learner updates beliefs correctly, but realized exposure is distorted before information is encountered. The aim is to identify when distorted exposure leaves learning robust and when it instead produces persistent ambiguity or convergence to an inferior source (informational lock-in).

## Research Question

Under what conditions is rational learning robust to distorted exposure, and when does exposure distortion generate persistent ambiguity or informational lock-in?

## Repository Structure

- `src/` — core model components, including the bandit environment, learning rules, exposure mechanism, metrics, and simulation logic
- `experiments/` — runnable scripts for parameter sweeps, diagnostics, robustness checks, and plotting
- `results/data/` — generated CSV outputs from simulation runs
- `results/figures/` — generated PDF figures used in the thesis

## Environment

Use Python 3.10 or higher.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
