# Federated Learning for IoT Intrusion Detection Using Relevance, Resource, and Communication

A research-grade Federated Learning platform built for edge IoT anomaly and intrusion detection using the **CICIoT2023** dataset, featuring the proposed **ARC** (Relevance, Resource, Communication) multi-criteria client evaluation and aggregation framework.

---

## 🏗️ Architecture

```
React Frontend (Port 5173)
       ↕ JWT-authenticated REST
Node.js + Express Backend (Port 5000)
       ↕ Database Queries        ↕ API-Key HTTP
PostgreSQL Database          Python FastAPI ML Service (Port 8000)
(Prisma ORM)                   ↕ PyTorch Aggregation
                          Dedicated Model Storage (filesystem)
```

---

## 🧠 ML Model

| Property          | Value                        |
|-------------------|------------------------------|
| Architecture      | `preact_resmlp_v3` (ResMLP)  |
| Input Features    | 45 network traffic features  |
| Output Classes    | 34 (IoT cyberattack/benign)  |
| Parameters        | ~4,981,628                   |
| Dataset           | CICIoT2023                   |
| Baseline Accuracy | 98.50% (on validation.csv)   |

---

## ⚡ ARC Framework (v1.0)

**Relevance:**
$$R_i = 0.60 Q_i + 0.40 D_i$$

**Resource:**
$$S_i = 0.40 M_i + 0.40 P_i + 0.20 A_i$$

**Communication:**
$$C_i = 0.50 B_i + 0.30 L_i + 0.20 T_i$$

**Overall ARC Score:**
$$\text{ARC}_i = 0.40 R_i + 0.30 S_i + 0.30 C_i$$

**ARC Aggregation Weight:**
$$\alpha_i = \frac{\text{ARC}_i}{\sum_{j=1}^K \text{ARC}_j}$$

All weights are **configurable** via the Admin dashboard and stored in the database.

---

## 📊 Dataset Setup & Requirements

### Repository Policy
No raw datasets, client traffic captures, or benchmark CSV files are tracked in this repository to prevent repository bloat and respect data distribution boundaries.

### How to Prepare Datasets
1. **Source:** Obtain the CICIoT2023 IoT network traffic dataset.
2. **Feature Specification:** Each CSV must contain the standard 45 numeric flow features plus a `label` column representing one of the 34 attack/benign traffic classes.
3. **File Placement:**
   - **Edge Client Partitions:** Save client-specific partitions into `storage/datasets/` (for example `client1_traffic.csv`, `client2_traffic.csv`, `client3_traffic.csv`) or pass any custom absolute path to the client agent via `--dataset`.
   - **Central Evaluation Benchmark:** Save your validation dataset (e.g. `validation.csv`) and update `VALIDATION_DATASET_PATH` in your `.env`.

### Operating Without Datasets
The application stack starts and functions cleanly without datasets:
- **Works Without Datasets:** Web dashboard, user authentication, training round creation, round lifecycle state management, ARC weight configuration, client enrollment, and system audit logs.
- **Requires Datasets:** Edge gradient training (`trainer.py` / `agent.py`), automated telemetry scoring ($Q_i, D_i$), and post-aggregation central model accuracy evaluation.

---

## 🚀 Quick Start

### Prerequisites
- Node.js v18+ / v20+ / v24+
- Python 3.10+ - 3.14+ (PyTorch, FastAPI)
- PostgreSQL 18 (default port: 5433 or 5432)

### 1. Clone & Setup Environment
```bash
cd arc-federated-learning
cp .env.example .env
# Edit .env with your database credentials and port (default: 5433)
```

### 2. Start PostgreSQL (if not running)
```powershell
# Windows: Start local cluster (if system service not running)
& "C:\Program Files\PostgreSQL\18\bin\pg_ctl.exe" -D ".\pgdata18" -l ".\pgdata18\pg.log" start
```

### 3. Initialize Database
```bash
cd backend
npm install
npx prisma db push
node prisma/seed.js
```

### 4. Start Backend
```bash
cd backend
node app.js
```

### 5. Start ML Service
```bash
cd ml-service
pip install -r requirements.txt
python run.py
```

### 6. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

For standalone edge node deployment, see [client-agent/README.md](./client-agent/README.md).

---

## 👤 Demo Accounts

| Role  | Email                   | Password   |
|-------|-------------------------|------------|
| Admin | admin@arc-fl.local      | Admin@123  |
| Client 1 | client1@arc-fl.local | Client@123 |
| Client 2 | client2@arc-fl.local | Client@123 |
| Client 3 | client3@arc-fl.local | Client@123 |

---

## 🔄 Federated Learning Workflow

1. **Admin** creates a training round.
2. **Clients** download the current global model.
3. **Clients** review and configure on-device training parameters (`epochs`, `batchSize`, `learningRate`) and ARC telemetry inputs (`qualityScore`, `diversityScore`, `availableRamGb`, `cpuIdlePct`, `bandwidthMbps`, `latencyMs`, `reliabilityPct`).
4. **Clients** train locally on their automatically assigned edge partition (`client1_traffic.csv`, etc.). Raw packet data never leaves the node.
5. **Clients** submit model updates (PyTorch state dict) with the preserved configuration snapshot.
6. **System** calculates official ARC scores using Admin-configured weights (anti-tampering protected).
7. **Admin** reviews submissions, selects clients, chooses aggregation strategy.
8. **Admin** previews weights → Merge Models.
9. **Python service** validates and aggregates parameters.
10. **System** evaluates the new global model on `validation.csv`.
11. **Admin** publishes the new global model version.

---

## 🎛️ Client Configuration & Telemetry Governance

| Configuration Parameter | Default Value | Source / Control | Editable by Client? | Description & Validation |
| :--- | :--- | :--- | :--- | :--- |
| **Local Epochs** | `3` | Client Manual | **Yes** | Positive integer ($1 \le E \le 50$) |
| **Batch Size** | `128` | Client Manual | **Yes** | Positive integer ($16 \le B \le 1024$) |
| **Learning Rate** | `0.001` | Client Manual | **Yes** | Positive float ($0.00001 \le \eta \le 1.0$) |
| **Data Quality ($Q_i$)** | Auto-measured | Dataset / Client | **Yes (Override)** | Measured from missing/inf ratios ($0 - 100\%$) |
| **Data Diversity ($D_i$)** | Auto-measured | Dataset / Client | **Yes (Override)** | Measured from Shannon entropy & class coverage ($0 - 100\%$) |
| **Available RAM** | Sensor-measured | `psutil` / Client | **Yes (Override)** | Physical free memory in GB ($> 0$) |
| **CPU Capacity** | Sensor-measured | `psutil` / Client | **Yes (Override)** | Idle processor percentage ($0 - 100\%$) |
| **Throughput** | Measured | PyTorch Trainer | **Yes (Override)** | Local processing rate in samples/second ($> 0$) |
| **Bandwidth** | Measured / Reported | Network / Client | **Yes (Override)** | Uplink transmission speed in Mbps ($> 0$) |
| **Network Latency** | Measured | Ping / RTT | **Yes (Override)** | Network round-trip delay in ms ($\ge 0$) |
| **Transmission Reliability** | Measured / Reported | Network / Client | **Yes (Override)** | Packet delivery rate ($0 - 100\%$) |
| **ARC Formula & Weights** | Admin configured | **Admin Only** | ❌ **No** | $\lambda_R, \lambda_S, \lambda_C$ strictly governed by Admin |
| **Assigned Dataset Partition** | System bound | **System Only** | ❌ **No** | Partition auto-bound to authenticated client node |

### Historical Snapshot Preservation
When training completes and the client submits their model update:
1. An immutable snapshot of the exact parameters used (`epochs`, `batchSize`, `learningRate`) is persisted with the `ClientSubmission` record.
2. The full telemetry breakdown, parameter source tags (`MEASURED` vs `MANUAL`), and the formula version used are archived in `ARCScore.rawTelemetryJson`.
3. Editing parameters for future rounds does **not** alter historical round submissions or previous ARC evaluations.

---

## 📁 Project Structure

```
arc-federated-learning/
├── frontend/          # React + Vite + Tailwind CSS
├── backend/           # Node.js + Express + Prisma
├── ml-service/        # Python FastAPI ML service
├── storage/           # Model & dataset artifacts
│   ├── global-models/ # Published immutable model packages
│   ├── client-updates/ # Edge client parameter uploads
│   ├── candidates/    # Aggregated candidates awaiting validation
│   └── datasets/      # Edge dataset partitions
├── docs/              # Architecture & API docs
├── pgdata/            # PostgreSQL local cluster (if used)
└── PROJECT_NOTES.md   # Central engineering reference
```

---

## 📚 Documentation

- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — System architecture
- [docs/ARC_FORMULA.md](./docs/ARC_FORMULA.md) — ARC scoring mathematics
- [docs/AGGREGATION.md](./docs/AGGREGATION.md) — Federated aggregation algorithms
- [docs/API.md](./docs/API.md) — REST API reference

---

## 🔒 Security Notes

- Raw edge datasets **never leave** client devices
- JWT authentication for all API calls
- Role-based access control (ADMIN vs CLIENT)
- Model binaries validated (SHA-256, key/shape compatibility) before aggregation
- API key authentication between Node.js backend and Python ML service
- Previous global model versions preserved — rollback available at any time

---

## ⚠️ Research Disclaimer

The ARC formula (v1.0) weights and sub-scoring definitions are **research proposals** subject to change. All weights are configurable via the Admin interface. Never claim validated research results without proper statistical evaluation.
