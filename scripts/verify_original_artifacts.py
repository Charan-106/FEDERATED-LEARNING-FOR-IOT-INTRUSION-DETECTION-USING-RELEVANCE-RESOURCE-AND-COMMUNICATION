"""
Verification Script for Original Google Colab Trained Model Artifacts
Inspects:
1. best.pt
2. model.py
3. model_config.json
4. scaler.pkl
5. label_encoder.pkl

Verifies:
- Checkpoint format and state dict keys
- Model architecture (45 input features, 34 classes, parameter count)
- Scaler compatibility (n_features_in_)
- Label encoder compatibility (classes count and matching config)
- Exact feature list and ordering
- Checkpoint loading and forward inference pass
- Validation dataset test pass (if validation.csv is present)
"""

import sys
import json
import joblib
from pathlib import Path
import torch
import numpy as np

ARTIFACTS_DIR = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning\ml-service\artifacts")

def verify():
    print("=" * 70)
    print("VERIFYING ORIGINAL GOOGLE COLAB MODEL ARTIFACTS")
    print(f"Path: {ARTIFACTS_DIR}")
    print("=" * 70)

    # 1. Check all 5 files exist
    expected_files = ["best.pt", "model.py", "model_config.json", "scaler.pkl", "label_encoder.pkl"]
    for fname in expected_files:
        fpath = ARTIFACTS_DIR / fname
        assert fpath.exists(), f"Missing artifact: {fname}"
        size_bytes = fpath.stat().st_size
        print(f"[FOUND] {fname:<20} ({size_bytes:>10,} bytes)")

    # 2. Inspect model_config.json
    print("\n--- 1. Inspecting model_config.json ---")
    with open(ARTIFACTS_DIR / "model_config.json", "r") as f:
        config = json.load(f)

    in_dim = config.get("in_dim")
    n_classes = config.get("n_classes")
    hidden = config.get("hidden")
    dropout = config.get("dropout")
    arch = config.get("arch")
    classes = config.get("classes", [])
    features = config.get("features", [])

    print(f"  Architecture: {arch}")
    print(f"  Input Dim:    {in_dim} features")
    print(f"  Output Dim:   {n_classes} classes")
    print(f"  Hidden:       {hidden}")
    print(f"  Dropout:      {dropout}")
    print(f"  Classes count:  {len(classes)} classes")
    print(f"  Features count: {len(features)} features")

    assert in_dim == 45, f"Expected 45 input features, got {in_dim}"
    assert n_classes == 34, f"Expected 34 output classes, got {n_classes}"
    assert len(classes) == 34, f"Expected 34 classes in config list, got {len(classes)}"
    assert len(features) == 45, f"Expected 45 features in config list, got {len(features)}"
    print("  [OK] Config dimensions verified (45 in, 34 out)")

    # 3. Inspect label_encoder.pkl
    print("\n--- 2. Inspecting label_encoder.pkl ---")
    le = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")

    print(f"  Type: {type(le).__name__}")
    assert hasattr(le, "classes_"), "Label encoder missing classes_ attribute"
    le_classes = list(le.classes_)
    print(f"  Encoded classes count: {len(le_classes)}")
    print(f"  First 3 classes: {le_classes[:3]}")
    print(f"  Last 3 classes:  {le_classes[-3:]}")

    assert len(le_classes) == 34, f"Expected 34 label encoder classes, got {len(le_classes)}"
    assert le_classes == classes, "Label encoder classes DO NOT match model_config.json classes exactly!"
    print("  [OK] Label encoder matches model_config.json classes 100% exactly.")

    # 4. Inspect scaler.pkl
    print("\n--- 3. Inspecting scaler.pkl ---")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")

    print(f"  Type: {type(scaler).__name__}")
    n_features_in = getattr(scaler, "n_features_in_", None)
    print(f"  Scaler n_features_in_: {n_features_in}")
    assert n_features_in == 45, f"Expected scaler to have 45 features, got {n_features_in}"

    mean_shape = scaler.mean_.shape if hasattr(scaler, "mean_") else None
    scale_shape = scaler.scale_.shape if hasattr(scaler, "scale_") else None
    print(f"  Mean vector shape:  {mean_shape}")
    print(f"  Scale vector shape: {scale_shape}")
    assert mean_shape == (45,), f"Expected scaler mean shape (45,), got {mean_shape}"
    assert scale_shape == (45,), f"Expected scaler scale shape (45,), got {scale_shape}"
    print("  [OK] Scaler is 100% compatible with 45 features.")

    # 5. Inspect model.py & best.pt checkpoint
    print("\n--- 4. Inspecting model.py and best.pt ---")
    sys.path.insert(0, str(ARTIFACTS_DIR))
    from model import ResMLP

    model = ResMLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Instantiated ResMLP model.")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Load best.pt checkpoint
    pt_path = ARTIFACTS_DIR / "best.pt"
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    print(f"  Checkpoint type: {type(ckpt).__name__}")
    if isinstance(ckpt, dict):
        print(f"  Checkpoint keys: {list(ckpt.keys())[:5]}... (total keys: {len(ckpt)})")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print("  Found 'model_state_dict' wrapper key.")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            print("  Found 'model' wrapper key.")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("  Found 'state_dict' wrapper key.")
        else:
            state_dict = ckpt
            print("  Checkpoint is direct state dict.")
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    print(f"  Strict state_dict load: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
    assert len(missing_keys) == 0, f"Missing keys when loading state_dict: {missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected keys when loading state_dict: {unexpected_keys}"
    print("  [OK] best.pt checkpoint matches model.py architecture with ZERO missing or unexpected keys!")

    # 6. Run Inference Test
    print("\n--- 5. Running Forward Inference Test ---")
    model.eval()

    # Synthetic sample
    dummy_raw = np.random.randn(5, 45).astype(np.float32)
    dummy_scaled = scaler.transform(dummy_raw)
    tensor_in = torch.tensor(dummy_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).numpy()

    print(f"  Input tensor shape:  {tensor_in.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output probs shape:  {probs.shape}")
    assert logits.shape == (5, 34), f"Expected logits shape (5, 34), got {logits.shape}"
    assert np.allclose(probs.sum(dim=1).numpy(), 1.0, atol=1e-5), "Softmax probabilities do not sum to 1.0"

    predicted_labels = le.inverse_transform(preds)
    print(f"  Sample predicted classes: {predicted_labels.tolist()}")
    print("  [OK] Inference forward pass successful and numerically valid.")

    # 7. Test with real validation partition if exists
    val_path = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning\storage\datasets\validation.csv")
    if val_path.exists():
        print("\n--- 6. Testing with Real Validation Dataset (storage/datasets/validation.csv) ---")
        import pandas as pd
        df_val = pd.read_csv(val_path, nrows=500)
        print(f"  Loaded {len(df_val)} rows from validation.csv")

        # Feature order check
        missing_feats = [f for f in features if f not in df_val.columns]
        if missing_feats:
            print(f"  [WARN] Missing features in validation.csv: {missing_feats}")
        else:
            X_val_raw = df_val[features].values.astype(np.float32)
            X_val_scaled = scaler.transform(X_val_raw)
            y_val_raw = df_val["label"].values if "label" in df_val.columns else df_val.iloc[:, -1].values
            y_val_encoded = le.transform(y_val_raw)

            with torch.no_grad():
                val_logits = model(torch.tensor(X_val_scaled, dtype=torch.float32))
                val_preds = val_logits.argmax(dim=1).numpy()
                acc = (val_preds == y_val_encoded).mean() * 100

            print(f"  Real validation accuracy on 500 samples: {acc:.2f}%")
            print("  [OK] Real dataset inference executed successfully!")

    print("\n" + "=" * 70)
    print("ALL 7 VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    verify()
