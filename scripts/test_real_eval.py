import json
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ARTIFACTS_DIR = Path("C:/Users/chara/.gemini/antigravity/scratch/arc-federated-learning/ml-service/artifacts")
sys.path.insert(0, str(ARTIFACTS_DIR))
from model import ResMLP

with open(ARTIFACTS_DIR / "model_config.json") as f:
    cfg = json.load(f)

le = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
model = ResMLP(in_dim=45, n_classes=34, hidden=cfg["hidden"], dropout=0.2)
ckpt = torch.load(ARTIFACTS_DIR / "best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()

features = cfg["features"]

for fname in ["benchmark_ciciot.csv", "client1_traffic.csv"]:
    csv_path = Path("C:/Users/chara/.gemini/antigravity/scratch/arc-federated-learning/storage/datasets") / fname
    df = pd.read_csv(csv_path)
    X_raw = df[features].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    y_raw = df["label"].values
    y_true = le.transform(y_raw)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        preds = logits.argmax(dim=1).numpy()

    acc = (preds == y_true).mean() * 100
    print(f"{fname:<25} ({len(df)} rows): Accuracy = {acc:.2f}%")
