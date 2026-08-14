# Hybrid DA-ML Algorithm for Emulating the Lorenz 96 Model

**Bachelor's Thesis — Nikolas Gavaz-Nicolae**  
Babeș-Bolyai University, Faculty of Mathematics and Computer Science  
Supervisor: Lect. Dr. Oana Lang | 2026

---

## Overview

This repository contains the implementation and experiments for combining **data assimilation (DA)** with **machine learning (ML)** to build a surrogate model of the chaotic Lorenz 96 system from sparse and noisy observations, following the hybrid iterative algorithm proposed by [Brajard et al. (2020)](https://doi.org/10.1016/j.jocs.2020.101171).

The surrogate model is a **residual convolutional neural network** that learns the system's dynamics without access to the governing equations — only from analysis fields produced by an Ensemble Kalman Filter.

## Method

At each iteration of the DA-ML cycle:

1. **DA step**: The Ensemble Kalman Filter (EnKF-N) is run with the current surrogate model, producing spatially complete analysis fields and covariance estimates.
2. **ML step**: The neural network is trained on the analysis fields, using the covariance matrix to weight the loss function.

The two steps reinforce each other: a better surrogate produces better analyses, and better analyses provide higher-quality training data.

## Experiments

Three configurations are compared, varying one component at a time:

| Configuration | Optimizer | Architecture | Parameters | RMSE-f (1 step) | RMSE-a | Λ₁ |
|---|---|---|---|---|---|---|
| **Reference** | Adagrad | Bilinear (4 layers) | 9,389 | 0.379 | 1.336 | 3.01 |
| **Experiment 1** | Adam | Bilinear (4 layers) | 9,389 | 0.283 | 1.220 | 2.73 |
| **Experiment 2** | Adagrad | Simplified (no bilinear) | 9,245 | 0.307 | 1.401 | 2.74 |

**Key findings:**
- Adam reduces RMSE-f by **25%** compared to Adagrad and converges twice as fast
- The bilinear layer does not provide a clear advantage on the reduced dataset (K=4,000)
- All configurations outperform cubic interpolation (RMSE-a: 1.2–1.4 vs. 2.44)

## Setup

| Parameter | Value |
|---|---|
| State variables (m) | 40 |
| Forcing (F) | 8 |
| Time step (h) | 0.05 |
| Observations per step (p) | 20 (50%) |
| Observation noise (σ_obs) | 1.0 |
| Ensemble size (N) | 30 |
| Model noise (σ_m) | 0.1 |
| DA-ML cycles | 30 |
| Epochs per cycle | 20 |
| Batch size | 256 |
| Dataset length (K) | 4,000 |

## Running

The notebook is designed to run on [Google Colab](https://colab.research.google.com/) with GPU acceleration.

1. Open `da_ml_lorenz96_experiments.ipynb` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → GPU)
3. Run all cells (Runtime → Run all)

Total runtime: ~3.3 hours on NVIDIA A100 for all 3 experiments.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib

All dependencies are available by default in Google Colab.

## Repository Structure

```
da-ml-lorenz96/
├── da_ml_lorenz96_experiments.ipynb   # Full implementation and experiments
└── README.md
```

## Reference

Brajard, J., Carrassi, A., Bocquet, M., Bertino, L.: *Combining data assimilation and machine learning to emulate a dynamical model from sparse and noisy observations: A case study with the Lorenz 96 model.* Journal of Computational Science, 44, 101171, 2020.

## License

This project was developed as part of a bachelor's thesis at Babeș-Bolyai University.
