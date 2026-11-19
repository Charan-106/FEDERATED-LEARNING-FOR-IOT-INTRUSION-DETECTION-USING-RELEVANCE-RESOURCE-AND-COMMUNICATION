# Federated Aggregation Algorithms
## ARC-FL Platform — Aggregation Reference

---

## 1. Prerequisites Before Aggregation

The system validates all selected submissions before executing aggregation:

| Check | Failure Action |
|---|---|
| Client eligibility (ACTIVE, ACCEPTED status) | Reject with reason |
| Model parameter key set matches reference | Reject with mismatch keys |
| Tensor shapes match reference architecture | Reject with shape diff |
| No NaN or Inf in parameter tensors | Reject with parameter name |
| Submission checksums match stored records | Reject with corruption notice |
| At least 1 client selected | Block aggregation |
| Weights sum to 1.0 (tolerance 1e-5) | Normalize or fail |

---

## 2. Aggregation Strategies

### Strategy 1: FedAvg (Sample-Weighted)

$$\alpha_i = \frac{n_i}{\sum_{j=1}^K n_j}$$

Where $n_i$ = number of local training samples on client $i$.

**Formula explanation:** Weights proportional to contribution size. Clients with larger local datasets receive higher influence. This is the classic McMahan et al. 2017 approach.

**Use case:** When data quality is assumed uniform across clients and quantity is the primary differentiator.

---

### Strategy 2: Equal-Weight Aggregation

$$\alpha_i = \frac{1}{K}$$

Where $K$ = number of selected clients.

**Formula explanation:** Every selected client contributes equally regardless of dataset size or performance metrics.

**Use case:** When fairness across clients is the primary objective, or when dataset sizes are known to be similar.

---

### Strategy 3: ARC-Weighted Aggregation *(Proposed)*

$$\alpha_i = \frac{\text{ARC}_i}{\sum_{j=1}^K \text{ARC}_j}$$

Where $\text{ARC}_i$ = composite multi-criteria score from the ARC framework.

**Formula explanation:** Clients with higher data quality, better hardware resources, and more reliable communication channels receive proportionally higher influence on the global model update.

**Use case:** Heterogeneous edge deployments where data quality, device capability, and network conditions vary significantly across the client fleet.

---

## 3. Parameter Averaging Mechanics

For all three strategies, the global model parameters are updated as:

$$\theta_{\text{global}} = \sum_{i=1}^K \alpha_i \cdot \theta_i$$

This applies to **all learnable parameters** (linear layer weights, biases).

### BatchNorm Handling (preact_resmlp_v3)

The ResMLP architecture uses BatchNorm1d extensively. Each BatchNorm layer contains:
- `weight` (gamma, learnable) → standard weighted average
- `bias` (beta, learnable) → standard weighted average
- `running_mean` → weighted float average
- `running_var` → weighted float average
- `num_batches_tracked` → integer **sum** (not average)

### Numerical Validation

Before aggregation, every tensor in every client submission is verified:
```python
if not torch.all(torch.isfinite(tensor)):
    raise AggregationError("NaN or Inf detected in parameter: {key}")
```

---

## 4. Model Compatibility Requirements

| Requirement | Description |
|---|---|
| Same architecture | All clients must submit state dicts compatible with `preact_resmlp_v3` |
| Exact key set | Parameter key names must exactly match the reference model |
| Exact tensor shapes | Each tensor's `.shape` must match the reference architecture |
| Same base model version | All submissions in a round must derive from the same global model version |
| Finite numerical values | No NaN or Inf permitted in any submitted parameter tensor |

**Missing parameters are NEVER treated as zeros.** A submission with missing keys is immediately rejected.

---

## 5. Candidate Model Flow

After aggregation, the result is a **candidate** model (not yet published):

1. Aggregated state dict saved to `storage/candidates/<candidate_id>/best.pt`
2. Supporting files copied from base model: `model.py`, `scaler.pkl`, `label_encoder.pkl`, `model_config.json`
3. SHA-256 checksum computed for the candidate checkpoint
4. Candidate metadata saved to `AggregationJob` record
5. Server-side evaluation on `validation.csv`
6. Admin reviews evaluation report
7. Admin explicitly publishes → creates new `GlobalModel` version

**Previous published models are never automatically overwritten.**
