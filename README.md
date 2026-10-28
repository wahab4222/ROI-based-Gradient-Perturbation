# ROI-Guided Gradient Perturbation in Federated Learning for Enhanced Alzheimer’s Disease Classification

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22094746.svg)](https://doi.org/10.5281/zenodo.22094746)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

This repository provides the official implementation and reproducibility assets accompanying the *SPIE Journal of Medical Imaging* (JMI) manuscript:
> **"ROI-Guided Gradient Perturbation in Federated Learning for Enhanced Alzheimer's Disease Classification"** (Manuscript ID: `JMI-260030P-2`).

The codebase supports **three-class AD / CN / MCI classification** on structural T1-weighted MRI from **OASIS** and **ADNI**, evaluating **EfficientNet-B2**, **Inception V3**, and **Swin-Tiny** under decentralized federated optimization. It includes non-perturbed federated learning, uniform Gaussian gradient perturbation, spatial ROI-guided perturbation (`ROI-GP`), layer-adaptive perturbation (`adaptive-GP`), layer-aware ROI perturbation (`ROI-layer-GP`), simulated secure aggregation, and comprehensive membership inference attack (MIA) privacy auditing.

**Archived Software Release:** [https://doi.org/10.5281/zenodo.22094746](https://doi.org/10.5281/zenodo.22094746)

---

## Important Privacy Notice and Methodological Scope

The methods implemented in this repository (`ROI-GP`, `adaptive-GP`, and `ROI-layer-GP`, along with legacy CLI flags `roi_dp`, `adaptive_dp`, `roi_layer_dp`) designate **empirical gradient perturbation strategies**.

**This implementation DOES NOT provide formal $(\epsilon, \delta)$-differential privacy guarantees:**
- **No Certified Accounting:** No Rényi Differential Privacy (RDP), Moments Accountant, Opacus accountant, or equivalent mathematical composition theorem is executed.
- **Model-Level Gradient Clipping:** Gradient clipping is applied to model/tensor gradient norms ($C = 1.0$), not sample-level microbatches as strictly mandated by standard DP-SGD.
- **Data-Dependent Heuristic:** Parameter sensitivity and spatial ROI importance maps are computed directly from training batch gradients prior to noise addition.
- **Empirical Evaluation Only:** Privacy resistance is evaluated empirically via confidence-, loss-, entropy-, and prediction-based membership inference attack AUC (MIA-AUC).

---

## Repository Structure

```text
ROI-Guided-Gradient-Perturbation/
├── unified_roi_dp.py             # Unified FL training, perturbation & evaluation pipeline
├── README.md                     # Project documentation and CLI guide
├── requirements.txt              # Environment dependencies
├── CITATION.cff                  # Machine-readable citation metadata
├── LICENSE                       # MIT open-source license
├── figures/
│   ├── fig1_fl_framework.png     # Federated perturbation setup diagram
│   └── fig6_qualitative_maps.png # Grad-CAM feature attribution & perturbation overlays
├── visualization/
│   ├── generate_qualitative_visualization.py
│   ├── experiment_summary.json
│   ├── final_metrics.csv
│   └── test_predictions.csv
└── legacy/
    ├── client.py                 # Historical Flower-based client script
    ├── server.py                 # Historical Flower-based server script
    └── README.md
