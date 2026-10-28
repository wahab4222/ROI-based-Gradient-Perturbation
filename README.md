# ROI-Guided Gradient Perturbation in Federated Learning for Enhanced Alzheimer’s Disease Classification

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22094746.svg)](https://doi.org/10.5281/zenodo.22094746)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

This repository provides the official implementation and supporting reproducibility assets for the *SPIE Journal of Medical Imaging* (JMI) manuscript:
> **"ROI-Guided Gradient Perturbation in Federated Learning for Enhanced Alzheimer's Disease Classification"** (Manuscript Reference: `JMI-260030P-2`).

The codebase supports **three-class AD / CN / MCI classification** from structural T1-weighted brain MRI scans on **OASIS** and **ADNI**, benchmarking convolutional backbones (**EfficientNet-B2**, **Inception V3**) and a transformer baseline (**Swin-Tiny**) within decentralized federated learning. 

Experimental conditions include non-perturbed federated learning, uniform Gaussian gradient perturbation, spatial ROI-guided perturbation (`ROI-GP`), layer-adaptive perturbation (`adaptive-GP`), layer-aware ROI perturbation (`ROI-layer-GP`), simulated secure aggregation (`ROI-GP+SA`), and membership inference attack (MIA) privacy risk auditing.

**Archived Software & Checkpoint Snapshot:** [https://doi.org/10.5281/zenodo.22094746](https://doi.org/10.5281/zenodo.22094746)

---

## Important Privacy Statement & Methodological Scope

The method identifiers in this repository (`ROI-GP`, `adaptive-GP`, `ROI-layer-GP`, as well as runtime CLI flags `roi_dp`, `adaptive_dp`, `roi_layer_dp`) designate **empirical gradient perturbation strategies**.

**The current implementation DOES NOT provide formally certified $(\epsilon, \delta)$-differential privacy:**
- **No Certified Accounting:** No Rényi Differential Privacy (RDP), Moments Accountant, Opacus accountant, or equivalent formal accountant is computed during training.
- **Model-Level Gradient Clipping:** Norm clipping is executed on model/tensor gradient norms ($C = 1.0$), rather than the sample-level microbatch clipping required by standard DP-SGD.
- **Data-Dependent Sensitivity:** Parameter sensitivity matrices and spatial ROI importance representations are computed directly from training gradients prior to perturbation.
- **Empirical Evaluation Only:** Privacy resistance is evaluated empirically through receiver operating characteristic analysis of membership inference attacks (MIA-AUC).

---

## Repository Structure

```text
ROI-Guided-Gradient-Perturbation/
├── unified_roi_dp.py             # Unified federated training, perturbation, and MIA evaluation
├── README.md                     # Comprehensive repository documentation
├── requirements.txt              # Environment dependencies
├── CITATION.cff                  # Machine-readable citation metadata
├── LICENSE                       # MIT Open-Source License
├── figures/
│   └── fig6_qualitative_maps.png # Grad-CAM feature attribution & noise scaling overlay
├── visualization/
│   ├── generate_qualitative_visualization.py  # Script reproducing Figure 6
│   ├── experiment_summary.json                # Run configuration metadata
│   ├── final_metrics.csv                      # Visualized run performance record
│   └── test_predictions.csv                   # Held-out predictions and probabilities
└── legacy/
    ├── client.py                 # Historical Flower-based client script
    ├── server.py                 # Historical Flower-based server script
    └── README.md
```

> **Note on Model Checkpoints:** The trained Inception V3 model checkpoint (`inceptionv3_roi_layer_beta2p0_best_val_global_model.pt`, ~83 MB) used for qualitative feature attribution is hosted in the immutable Zenodo archive ([DOI: 10.5281/zenodo.22094746](https://doi.org/10.5281/zenodo.22094746)) to maintain a lightweight repository footprint.

---

## Main Execution Script

All federated training runs, perturbation variants, and membership inference evaluations are managed by the unified runner:

```bash
python unified_roi_dp.py --help
```

### Command-Line Arguments

| Argument | Format / Options | Default | Description |
|---|---|---|---|
| `--dataset` | `oasis`, `adni` | `oasis` | Target clinical neuroimaging cohort. |
| `--data_root` | `/path/to/dataset` | *Required* | Root directory containing train/val/test splits. |
| `--backbone` | `efficientnet_b2`, `inception_v3`, `swin_t` | `inception_v3` | Neural network feature extractor backbone. |
| `--mode` | `fed_nondp`, `fed_dp`, `roi_dp`, `adaptive_dp`, `roi_layer_dp`, `roi_dp_sa` | `roi_dp` | Perturbation and federated aggregation mode. |
| `--noise_multiplier` | Float (e.g., `0.0020`, `0.0008`) | `0.0020` | Base Gaussian perturbation scale $\sigma$. |
| `--match_rms` | Flag (`--match_rms`) | Disabled | Enforces tensor root-mean-square normalization ($\mathbb{E}[\tilde{S}^2]=1.0$). |
| `--eval_mode` | `base`, `tta`, `auto` | `base` | Deterministic center cropping (`base`) vs. 3-view test-time augmentation (`tta`). |
| `--seed` | Integer (`42`, `43`, `44`) | `42` | Random seed for initialization and client partitioning. |
| `--rounds` | Integer | `30` | Number of federated communication rounds. |
| `--clients` | Integer | `6` | Number of simulated federated institutional clients. |
| `--local_epochs` | Integer | `3` | Local training epochs per client per round. |
| `--spatial_beta` | Float (`0.5`, `1.0`, `2.0`) | `2.0` | Spatial strength coefficient $\beta$ for `ROI-layer-GP`. |

---

## Supported Datasets

1. **OASIS** (Open Access Series of Imaging Studies)
2. **ADNI** (Alzheimer's Disease Neuroimaging Initiative)

### Diagnostic Categories
```text
AD   (Alzheimer's Disease)
CN   (Cognitively Normal)
MCI  (Mild Cognitive Impairment)
```

### Expected Directory Layout
```text
dataset_root/
├── train/
│   ├── AD/
│   ├── CN/
│   └── MCI/
├── val/
│   ├── AD/
│   ├── CN/
│   └── MCI/
└── test/
    ├── AD/
    ├── CN/
    └── MCI/
```

*Raw structural MRI scans are not redistributed with this code. Users must request and obtain the datasets directly through the official OASIS Brains and ADNI/LONI portals in compliance with their respective Data Use Agreements.*

---

## Supported Architectures

| CLI Backbone Identifier | Neural Architecture | Native Resolution | Parameter Dimensions |
|---|---|---:|---|
| `efficientnet_b2` | EfficientNet-B2 | $260 \times 260$ | Compact depthwise-separable convs |
| `inception_v3` | Inception V3 | $299 \times 299$ | Deep multi-branch convolutional blocks |
| `swin_t` | Swin-Tiny | $224 \times 224$ | Hierarchical shifted-window Transformer |

*The spatial ROI mechanism operates on 4D convolutional weight tensors ($C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w$). Swin-Tiny serves as an architectural generalization baseline without direct 4D kernel scaling.*

---

## Experimental Perturbation Modes

- **`fed_nondp`**: Standard federated learning without noise injection (utility upper bound).
- **`fed_dp`**: Spatially and layer-wise uniform Gaussian gradient perturbation.
- **`roi_dp` (`ROI-GP`)**: Spatial/kernel-guided gradient perturbation targeting 4D convolutional weight tensors.
- **`adaptive_dp` (`adaptive-GP`)**: Layer-adaptive perturbation scaling noise by tensor-level gradient norms.
- **`roi_layer_dp` (`ROI-layer-GP`)**: Integrated formulation combining layer-level sensitivity with spatial 4D filter modulation.
- **`roi_dp_sa`**: ROI-guided perturbation combined with simulated secure aggregation constraints.

---

## Federated Optimization Protocol

- **Aggregation:** Synchronous, round-based federated learning with sample-weighted **FedAvg**:
  $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{\sum_j n_j} w_{t+1}^{(k)}$$
- **Optimizer:** AdamW with weight decay $7 \times 10^{-5}$ and classifier dropout 0.30.
- **Training Schedule:** Communication rounds 1–2 execute classifier-head warm-up ($8 \times 10^{-4}$); subsequent rounds fine-tune the classification head ($3 \times 10^{-4}$) and designated deep backbone blocks ($2 \times 10^{-5}$).
- **Client Partitioning:** Image-level, class-stratified allocation across simulated institutions.

---

## Reproducing the Matched Multi-Seed Benchmark (Section 4.11 / Table 5)

To reproduce the controlled benchmark reported in **Table 5** of the manuscript, run all arms with an identical base noise scale ($\sigma = 0.0020$), tensor-level RMS normalization (`--match_rms`), deterministic evaluation (`--eval_mode base`), and validation-only checkpoint selection across seeds 42, 43, and 44:

```bash
# 1. Uniform Perturbation Baseline (Matched RMS)
python unified_roi_dp.py \
  --dataset oasis --backbone inception_v3 --mode fed_dp \
  --noise_multiplier 0.0020 --match_rms --eval_mode base --seed 42 \
  --data_root /path/to/oasis

# 2. Gradient-Guided Perturbation (ROI-GP)
python unified_roi_dp.py \
  --dataset oasis --backbone inception_v3 --mode roi_dp \
  --noise_multiplier 0.0020 --match_rms --eval_mode base --seed 42 \
  --data_root /path/to/oasis

# 3. Layer-Adaptive Perturbation (adaptive-GP)
python unified_roi_dp.py \
  --dataset oasis --backbone inception_v3 --mode adaptive_dp \
  --noise_multiplier 0.0020 --match_rms --eval_mode base --seed 42 \
  --data_root /path/to/oasis
```

*(Repeat the commands replacing `--seed 42` with `--seed 43` and `--seed 44` to reproduce the aggregate statistics and paired Student-$t$ significance values).*

---

## Membership Inference Attack (MIA) Evaluation

Following federated optimization, held-out privacy leakage is audited by attacking the global model with five sample-level score metrics:
1. True-class predicted probability: $s_{\text{TC}}(x_i) = p_\theta(y_i \mid x_i)$ (Primary MIA metric)
2. Maximum predicted output confidence
3. Negative predictive Shannon entropy
4. Negative cross-entropy classification loss
5. Binary classification correctness

The pipeline calculates the area under the ROC curve (MIA-AUC):
- $\text{MIA-AUC} \approx 0.50$: Near-random separability between training members and non-members.
- Higher $\text{MIA-AUC}$: Higher membership distinguishability and empirical privacy leakage.

---

## Qualitative Visualization (Figure 6)

The multi-panel Grad-CAM feature attribution and perturbation scaling visualization in the manuscript corresponds to:
- **Cohort:** OASIS
- **Backbone:** Inception V3
- **Perturbation Mode:** `roi_layer_dp` ($\beta = 2.0$, Random Seed 42)

### Reproduction Steps:
1. Download the trained model checkpoint `inceptionv3_roi_layer_beta2p0_best_val_global_model.pt` from [Zenodo](https://doi.org/10.5281/zenodo.22094746).
2. Place the `.pt` file inside the `visualization/` directory (or update `CKPT_PATH` in the script).
3. Execute:
   ```bash
   python visualization/generate_qualitative_visualization.py \
     --data_root /path/to/oasis \
     --checkpoint visualization/inceptionv3_roi_layer_beta2p0_best_val_global_model.pt \
     --output figures/fig6_qualitative_maps.png
   ```

The script generates four columns:
1. Input structural MRI scan;
2. Grad-CAM feature importance map extracted from block `Mixed_7c`;
3. Derived spatial perturbation-scale map;
4. Anatomical feature overlay.

---

## Installation & Setup

Python 3.9 or higher is recommended:

```bash
# Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*Install the PyTorch wheel compatible with your local CUDA driver environment from [pytorch.org](https://pytorch.org/).*

---

## Reproducibility and Versioning Note

The manuscript presents both historical exploratory baseline packages (Tables 1–4) and the definitive matched-noise multi-seed benchmark (Table 5, Section 4.11). Some historical runs differ in client counts, training rounds, or evaluation modes (documented transparently in the text). Readers and reviewers are directed to Section 4.11 for the strictly controlled comparative evaluation.

Archived Zenodo release: **[https://doi.org/10.5281/zenodo.22094746](https://doi.org/10.5281/zenodo.22094746)**

---

## Legacy Implementation

Earlier iterations of this codebase used decentralized Flower `client.py` and `server.py` scripts targeting four-class OASIS classification with CBAM attention and FedProx aggregation. These files have been moved to `legacy/` for historical provenance and are superseded by `unified_roi_dp.py`.

---

## Citation

If you use this codebase, methodology, or experimental artifacts, please cite:

```bibtex
@article{wahab2026roi,
  title   = {ROI-Guided Gradient Perturbation in Federated Learning for Enhanced Alzheimer's Disease Classification},
  author  = {Wahab, Abdul and Wang, Yanyan and Mehmood, Muhammad Hamza and Ahmad, Talha Shahan},
  journal = {Journal of Medical Imaging},
  year    = {2026},
  note    = {Under revision, Manuscript JMI-260030P-2}
}

@software{wahab_zenodo_2026,
  author    = {Wahab, Abdul},
  title     = {ROI-Guided Gradient Perturbation in Federated Learning for Enhanced Alzheimer's Disease Classification},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22094746},
  url       = {https://doi.org/10.5281/zenodo.22094746}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## License

This software is released under the **MIT License**. See [`LICENSE`](LICENSE) for terms.
