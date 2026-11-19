# ARC-FL REST API Specification
## Formal Reference Manual

Base URL: `http://localhost:5000/api`

All protected endpoints require an `Authorization: Bearer <token>` HTTP header.

---

## 1. Authentication Endpoints

### 1.1 `POST /auth/register`
Register a new user account.
- **Access:** Public
- **Request Body:**
  ```json
  {
    "email": "user@arc-fl.local",
    "password": "Password@123",
    "name": "Edge Node Delta",
    "role": "CLIENT", // "ADMIN" or "CLIENT"
    "clientId": "client-004", // required if role is CLIENT
    "deviceType": "Raspberry Pi 4",
    "hardwareSpecs": { "cpu": "ARM Cortex-A72", "ram_gb": 4.0 }
  }
  ```
- **Response (201):** User object and signed JWT token.

### 1.2 `POST /auth/login`
Authenticate existing user.
- **Access:** Public
- **Request Body:**
  ```json
  { "email": "admin@arc-fl.local", "password": "Admin@123" }
  ```
- **Response (200):**
  ```json
  {
    "token": "eyJhbGciOiJIUzI1NiIsIn...",
    "user": { "id": "...", "email": "admin@arc-fl.local", "name": "System Administrator", "role": "ADMIN" }
  }
  ```

### 1.3 `GET /auth/me`
Retrieve profile of currently authenticated user.
- **Access:** Authenticated

---

## 2. Client Endpoints

### 2.1 `GET /client/model`
Retrieve metadata of the currently active global model.
- **Access:** CLIENT role

### 2.2 `GET /client/model/download`
Stream the physical `best.pt` file of the active global model.
- **Access:** CLIENT role
- **Headers Returned:** `Content-Disposition: attachment; filename="model-<version>.pt"`

### 2.3 `GET /client/submissions`
List all training updates submitted by this client.
- **Access:** CLIENT role

### 2.4 `GET /client/arc-history`
List historical ARC score evaluations for this client.
- **Access:** CLIENT role

### 2.5 `GET /client/datasets`
List permitted local edge traffic CSV datasets available on the device.
- **Access:** CLIENT role

### 2.6 `POST /client/train`
Execute local on-device PyTorch training on a private CSV partition.
- **Access:** CLIENT role
- **Client Editable Parameters:**
  - `epochs` (Integer, 1 to 50): Number of local training epochs.
  - `batchSize` (Integer, 16 to 1024): Mini-batch sample size.
  - `learningRate` (Float, 0.00001 to 1.0): Optimizer step size.
- **Request Body:**
  ```json
  {
    "epochs": 3,
    "batchSize": 64,
    "learningRate": 0.002
  }
  ```
- **Response (200):**
  ```json
  {
    "accuracy": 84.0,
    "loss": 0.5801,
    "samples_trained": 600,
    "epochs": 3,
    "batch_size": 64,
    "learning_rate": 0.002,
    "training_duration_seconds": 1.09,
    "update_id": "3807bcf5-4344-456f-98ef-64c18b0f6774",
    "update_path": "client-updates/3807bcf5-4344-456f-98ef-64c18b0f6774.pt",
    "resource_metrics": {
      "available_ram_gb": 4.65,
      "total_ram_gb": 16.0,
      "cpu_idle_pct": 29.6,
      "throughput_samples_per_sec": 1098.1
    },
    "quality_metrics": { "missing_ratio": 0.0, "inf_ratio": 0.0, "preprocessing_success": 1.0 },
    "diversity_metrics": { "class_count": 23, "entropy": 3.742, "minority_class_coverage": 0.676 }
  }
  ```

### 2.7 `POST /client/submit-trained`
Submits trained PyTorch checkpoint update (`.pt`) alongside client-provided parameters, measured telemetry, and client overrides.
- **Access:** CLIENT role
- **Field Categorization & Governance:**
  | Field | Category | Governance | Description |
  | :--- | :--- | :--- | :--- |
  | `epochs`, `batchSize`, `learningRate` | **Client Editable** | Validated (ranges enforced) | Training hyperparameters configured by client. |
  | `qualityScore`, `diversityScore` | **Client Editable / Auto-Measured** | Client Override or Dataset Measured | Relevance metrics; defaults to dataset calculation. |
  | `availableRamGb`, `cpuIdlePct` | **Client Editable / Auto-Measured** | Client Override or Sensor Measured | Hardware metrics; defaults to `psutil` sensor capture. |
  | `throughputSamplesPerSec` | **Auto-Measured / Client Editable** | Training Throughput | Samples processed per second during training. |
  | `bandwidthMbps`, `latencyMs`, `reliabilityPct` | **Client Editable / Auto-Measured** | Client Override or Network Measured | Network communication telemetry. |
  | `overallArcScore`, Component Scores | **System-Calculated** | **Admin Controlled** | Calculated strictly on server using official ARC formula weights. |
  | Formula Weights ($\lambda_R, \lambda_S, \lambda_C$) | **Admin Exclusive** | **Admin Only** (`PUT /api/arc/config`) | Formula weights cannot be altered by clients. |
- **Historical Snapshot Preservation:**
  - Upon submission, `ClientSubmission` records the exact snapshot of `epochs`, `batchSize`, `learningRate`, `storagePath`, and SHA256 checksum.
  - `ARCScore` preserves the complete telemetry snapshot, formula version, and weights used in `rawTelemetryJson`.
  - Future edits to client configuration do not modify past round submissions or scores.
- **Request Body:**
  ```json
  {
    "roundId": "e0d69762-aec8-458f-b5bb-1b1f7d80081c",
    "updatePath": "client-updates/3807bcf5-4344-456f-98ef-64c18b0f6774.pt",
    "localAccuracy": 84.0,
    "localLoss": 0.5801,
    "samplesTrained": 600,
    "epochs": 3,
    "batchSize": 64,
    "learningRate": 0.002,
    "trainingDurationSeconds": 1.09,
    "resourceMetrics": { "available_ram_gb": 12.5, "total_ram_gb": 32.0, "cpu_idle_pct": 75.0, "throughput_samples_per_sec": 850.0 },
    "qualityMetrics": { "quality_score": 99.2, "missing_ratio": 0.0, "inf_ratio": 0.0, "preprocessing_success": 1.0 },
    "diversityMetrics": { "diversity_score": 88.5, "class_count": 22, "entropy": 3.65, "minority_class_coverage": 0.65 },
    "commMetrics": { "bandwidth_mbps": 75.0, "latency_ms": 25.0, "reliability_pct": 99.8 },
    "parameterSources": { "epochs": "MANUAL", "batchSize": "MANUAL", "learningRate": "MANUAL", "qualityScore": "MANUAL", "availableRamGb": "MANUAL" }
  }
  ```
- **Response (201):** Submission metadata; triggers async ARC evaluation using official Admin formula.

---

## 3. Administrative Endpoints

### 3.1 `GET /admin/dashboard`
Summary KPIs and overview telemetry.
- **Access:** ADMIN role

### 3.2 `GET /admin/clients`
List all registered edge clients with hardware specs and last active timestamp.
- **Access:** ADMIN role

### 3.3 `PATCH /admin/clients/:clientId/status`
Activate or deactivate an edge client node.
- **Access:** ADMIN role
- **Request Body:** `{ "status": "INACTIVE" }` // "ACTIVE" or "INACTIVE"

### 3.4 `GET /admin/rounds`
List all federated training rounds.
- **Access:** ADMIN role

### 3.5 `POST /admin/rounds`
Initialize a new federated training round using the active global model.
- **Access:** ADMIN role

### 3.6 `GET /admin/rounds/:roundId/submissions`
List all client submissions under a specific training round.
- **Access:** ADMIN role

### 3.7 `GET /admin/submissions`
List submissions fleet-wide with optional filters (`roundId`, `status`).
- **Access:** ADMIN role

### 3.8 `PATCH /admin/submissions/:submissionId/review`
Accept or reject a client submission.
- **Access:** ADMIN role
- **Request Body:** `{ "status": "ACCEPTED", "validationMessage": "Verified" }`

### 3.9 `GET /admin/analytics`
Retrieve round-by-round convergence trajectories, average client accuracy, and loss.
- **Access:** ADMIN role

### 3.10 `GET /admin/audit-logs`
Retrieve administrative activity logs.
- **Access:** ADMIN role

---

## 4. ARC Framework Endpoints

### 4.1 `GET /arc/scores`
Query calculated ARC scores (filter by `roundId`).
- **Access:** Authenticated

### 4.2 `GET /arc/config`
Retrieve current active ARC formula configuration and weights.
- **Access:** Authenticated

### 4.3 `GET /arc/configs`
Retrieve history of all past formula versions and parameter changes.
- **Access:** ADMIN role

### 4.4 `PUT /arc/config`
Update dynamic formula weights with $\sum \lambda = 1.0$ validation.
- **Access:** ADMIN role
- **Request Body:**
  ```json
  {
    "lambdaRelevance": 0.40,
    "lambdaResource": 0.30,
    "lambdaCommunication": 0.30,
    "weightQuality": 0.60,
    "weightDiversity": 0.40,
    "weightMemory": 0.40,
    "weightProcessing": 0.40,
    "weightCapacity": 0.20,
    "weightBandwidth": 0.50,
    "weightLatency": 0.30,
    "weightReliability": 0.20,
    "recalculatePending": true
  }
  ```

### 4.5 `POST /arc/recalculate`
Explicitly recalculate ARC scores for pending submissions using the active weights.
- **Access:** ADMIN role
- **Request Body:** `{ "roundId": "optional-round-id" }`

---

## 5. Aggregation Endpoints

### 5.1 `POST /aggregation/preview`
Calculate and preview client aggregation weights before committing to a merge.
- **Access:** ADMIN role
- **Request Body:**
  ```json
  {
    "roundId": "e0d69762-aec8-458f-b5bb-1b1f7d80081c",
    "submissionIds": ["sub-1", "sub-2", "sub-3"],
    "strategy": "ARC_WEIGHTED" // "FEDAVG", "EQUAL", or "ARC_WEIGHTED"
  }
  ```

### 5.2 `POST /aggregation/merge`
Execute federated averaging on the Python ML service and evaluate candidate on `validation.csv`.
- **Access:** ADMIN role
- **Request Body:** Same as preview
- **Response (200):**
  ```json
  {
    "message": "Aggregation completed. Candidate model evaluated.",
    "jobId": "bd8c779f-18ab-4be8-baec-f66e5b898edc",
    "candidate_id": "1bef0190-9bbe-46a4-82fe-521a7d1798c7",
    "candidatePath": "storage/candidates/1bef0190-9bbe-46a4-82fe-521a7d1798c7.pt",
    "candidateChecksum": "2bf696658f376f0c39571dc3...",
    "strategy": "ARC_WEIGHTED",
    "validClientCount": 3,
    "weights": [ ... ],
    "evaluation": {
      "accuracy": 72.04,
      "macro_f1": 53.68,
      "macro_precision": 62.67,
      "macro_recall": 58.67,
      "loss": 1.6185,
      "sample_count": 5000
    }
  }
  ```

### 5.3 `GET /aggregation/history`
Query all past aggregation jobs, applied strategies, and participant weights.
- **Access:** ADMIN role

---

## 6. Global Model Endpoints

### 6.1 `GET /models/current`
Get currently active global model metadata.
- **Access:** Authenticated

### 6.2 `GET /models/versions`
List all published immutable model versions.
- **Access:** ADMIN role

### 6.3 `GET /models/:modelId/download`
Stream the physical `best.pt` file of any historical model version.
- **Access:** Authenticated

### 6.4 `POST /models/publish-candidate`
Package and publish an aggregated candidate model as the new active global release.
- **Access:** ADMIN role
- **Request Body:**
  ```json
  {
    "roundId": "e0d69762-aec8-458f-b5bb-1b1f7d80081c",
    "jobId": "bd8c779f-18ab-4be8-baec-f66e5b898edc",
    "version": "global-v2",
    "name": "ARC-FL IoT IDS Model v2"
  }
  ```

### 6.5 `POST /models/:modelId/rollback`
Instantly roll back the active global model pointer to a previous validated version.
- **Access:** ADMIN role
