# ARC-FL Remote Client Agent

Standalone, edge-deployable Federated Learning Client for IoT Intrusion Detection on the ARC-FL platform.

## Overview
The `client_agent.py` script is designed to run independently on remote client hardware (e.g. Raspberry Pi, Jetson Nano, edge industrial PC, or remote cloud VM). It communicates exclusively with the central ARC-FL server over HTTP/HTTPS REST APIs.

### Security & Privacy Guarantees
- **Zero Raw Data Upload:** Client datasets never leave the local device. Only trained PyTorch weight updates and high-level statistical telemetry are uploaded.
- **Mutual Integrity Verification:** Base model checkpoints are verified via SHA-256 before training starts. Updates are signed and verified by the server.
- **Tenant Isolation:** Client actions are authenticated via JWT Bearer tokens tied to specific client credentials. Clients cannot access or alter other clients' submissions.

---

## Installation & Requirements

On the edge client device:
```bash
# Recommended Python version: 3.9 - 3.12
pip install torch pandas numpy requests scikit-learn joblib
```

---

## Usage

```bash
python agent.py \
  --server https://arc-fl.yourdomain.com \
  --email client1@arc-fl.local \
  --password "YourClientPassword" \
  --dataset /path/to/private/traffic_partition.csv \
  --workdir ./workspace \
  --scaler /path/to/scaler.pkl \
  --encoder /path/to/label_encoder.pkl
```

### Command-Line Arguments
| Parameter | Required | Description |
| :--- | :---: | :--- |
| `--server` | No | Central ARC-FL backend URL (default: `http://localhost:5000`) |
| `--email` | Yes | Registered client email |
| `--password` | Yes | Client password |
| `--dataset` | Yes | Absolute path to local private network traffic CSV |
| `--workdir` | No | Directory for downloading checkpoints and storing model updates |
| `--scaler` | No | Optional path to `StandardScaler.pkl` |
| `--encoder` | No | Optional path to `LabelEncoder.pkl` |

---

## Production Central Server Deployment Requirements

For deploying the central ARC-FL services in production:

1. **Reverse Proxy & HTTPS:**
   - Terminate SSL/TLS (Let's Encrypt / Certbot) on Nginx or Caddy.
   - Reverse proxy port 443 to Node.js backend port 5000.
   - Keep internal Python ML service (port 8000) and PostgreSQL (port 5432) strictly bound to `127.0.0.1` (never expose ports 8000 or 5432 to public internet).
2. **Secrets & Credentials:**
   - Replace development `JWT_SECRET` and `ML_SERVICE_API_KEY` with cryptographically random 256-bit secrets.
3. **Database & Storage Backups:**
   - Configure automated daily `pg_dump` backups with offsite replication.
   - Retain immutable snapshots of `storage/global-models/initial-v1/best.pt`.
