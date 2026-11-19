# ARC Formula — Mathematical Reference
## ARC Federated Learning Platform — Version 1.0

This document describes the mathematical formulation of the **ARC (Relevance, Resource, Communication)** multi-criteria client evaluation framework.

---

## 1. Overview

In standard Federated Averaging (FedAvg), client contributions are weighted solely by sample count:

$$\alpha_i^{\text{FedAvg}} = \frac{n_i}{\sum_{j=1}^K n_j}$$

This ignores three critical dimensions in heterogeneous edge deployments:
- **Relevance** — data quality and class diversity of the edge partition
- **Resource** — available hardware capacity on the edge device
- **Communication** — network performance characteristics

ARC computes a composite importance score $\text{ARC}_i$ that integrates all three dimensions.

---

## 2. Component Scores (All normalized to [0, 100])

### 2.1 Relevance Score ($R_i$)

$$R_i = w_Q \cdot Q_i + w_D \cdot D_i$$

**Default weights:** $w_Q = 0.60$, $w_D = 0.40$

| Sub-Component | Variable | Description | Normalization |
|---|---|---|---|
| Data Quality | $Q_i$ | Missing/invalid feature ratio, preprocessing success | $Q_i = \max(0, \min(100, (1 - \text{missing\_ratio}) \times 100))$ |
| Data Diversity | $D_i$ | Label entropy and minority class coverage | $D_i = \frac{H(\mathbf{y}_i)}{H_{\max}} \times 100$ |

**Privacy Guarantee:** $Q_i$ and $D_i$ are computed **locally on the edge device** from non-sensitive summary statistics (missing counts, label histograms). Raw feature vectors are never transmitted.

---

### 2.2 Resource Score ($S_i$)

$$S_i = w_M \cdot M_i + w_P \cdot P_i + w_A \cdot A_i$$

**Default weights:** $w_M = 0.40$, $w_P = 0.40$, $w_A = 0.20$

| Sub-Component | Variable | Description | Normalization |
|---|---|---|---|
| Memory | $M_i$ | Available RAM headroom | $M_i = \max(0, \min(100, (1 - \text{ram\_used\_ratio}) \times 100))$ |
| Processing | $P_i$ | CPU idle capacity | $P_i = \max(0, \min(100, \text{cpu\_idle\_pct}))$ |
| Computing Capacity | $A_i$ | Training throughput ratio | $A_i = \max(0, \min(100, (\text{throughput} / \text{target}) \times 100))$ |

**Note:** The scoring does not assume GPU availability. CPU-only devices are scored on CPU idle percentage and achievable batch throughput.

---

### 2.3 Communication Score ($C_i$)

$$C_i = w_B \cdot B_i + w_L \cdot L_i + w_T \cdot T_i$$

**Default weights:** $w_B = 0.50$, $w_L = 0.30$, $w_T = 0.20$

| Sub-Component | Variable | Description | Normalization |
|---|---|---|---|
| Bandwidth | $B_i$ | Upload throughput | $B_i = \min(100, (\text{Mbps} / 10.0) \times 100)$ |
| Latency | $L_i$ | Round-trip latency | $L_i = \max(0, (1 - \text{latency\_ms} / 500) \times 100)$ |
| Reliability | $T_i$ | Successful transmission ratio | $T_i = \text{reliability\_pct}$ (measured as %) |

**Directional Guarantees:**
- Higher bandwidth → Higher $B_i$ ✓
- **Lower latency → Higher $L_i$** ✓ (inverted: 0ms = 100 pts, 500ms = 0 pts)
- Higher reliability → Higher $T_i$ ✓

---

## 3. Overall ARC Score

$$\text{ARC}_i = \lambda_R \cdot R_i + \lambda_S \cdot S_i + \lambda_C \cdot C_i$$

**Default weights (v1.0):**

| Weight | Symbol | Default | Meaning |
|---|---|---|---|
| Relevance | $\lambda_R$ | 0.40 | Importance of data quality and diversity |
| Resource | $\lambda_S$ | 0.30 | Importance of hardware capability |
| Communication | $\lambda_C$ | 0.30 | Importance of network performance |

**Constraint:** $\lambda_R + \lambda_S + \lambda_C = 1.0$

---

## 4. Aggregation Weight Calculation

### 4.1 ARC-Weighted Aggregation

$$\alpha_i^{\text{ARC}} = \frac{\text{ARC}_i}{\sum_{j=1}^K \text{ARC}_j}$$

### 4.2 FedAvg (Sample-Weighted)

$$\alpha_i^{\text{FedAvg}} = \frac{n_i}{\sum_{j=1}^K n_j}$$

### 4.3 Equal-Weight

$$\alpha_i^{\text{Equal}} = \frac{1}{K}$$

**Normalization Guarantee:** For all strategies: $\sum_{i=1}^K \alpha_i = 1.0$

---

## 5. Federated Parameter Aggregation

$$W_{\text{global}} = \sum_{i=1}^K \alpha_i \cdot W_i$$

### BatchNorm Policy:
| Parameter Type | Aggregation Rule |
|---|---|
| Learnable weights & biases | Weighted float average: $\sum \alpha_i \theta_i$ |
| BatchNorm running_mean, running_var | Weighted float average: $\sum \alpha_i \mu_i$ |
| BatchNorm num_batches_tracked | Integer sum: $\sum_i \text{batches}_i$ |

---

## 6. Configuration

All weights ($\lambda_R, \lambda_S, \lambda_C$, sub-weights $w_Q, w_D$, etc.) are stored in the `ARCConfig` database table and are **dynamically configurable** via the Admin dashboard without code changes.

**Formula Version:** `1.0` (stored per ARCScore record for reproducibility)

---

## 7. Research Notes

- The formula is Version 1.0 and subject to experimental refinement.
- Weights represent **research proposals**, not validated optimal values.
- Experiments should compare ARC-weighted aggregation against FedAvg baseline across multiple rounds.
- Future versions may incorporate gradient divergence or loss surface curvature metrics into Relevance scoring.
