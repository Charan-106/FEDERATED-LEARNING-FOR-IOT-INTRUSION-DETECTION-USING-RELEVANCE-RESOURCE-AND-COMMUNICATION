"""
Dataset Provenance & Verification Investigation Script
Investigates:
1. Exact source and slice origin of client1_traffic.csv, client2_traffic.csv, client3_traffic.csv
2. Matching against P:/project/train.csv, validation.csv, test.csv
3. Sample selection method and independence
4. Class distributions of client1 and client2
5. Feature compatibility with 45 features, scaler.pkl, and label_encoder.pkl
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

DATASETS_DIR = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning\storage\datasets")
PROJECT_DIR = Path(r"P:\project")
ARTIFACTS_DIR = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning\ml-service\artifacts")

def main():
    print("=" * 70)
    print("INVESTIGATING DATASET PROVENANCE")
    print("=" * 70)

    # 1. Inspect client1_traffic.csv and client2_traffic.csv
    c1_path = DATASETS_DIR / "client1_traffic.csv"
    c2_path = DATASETS_DIR / "client2_traffic.csv"
    c3_path = DATASETS_DIR / "client3_traffic.csv"

    print(f"\n[1] Client dataset files existence:")
    print(f"    client1: {c1_path.exists()} ({c1_path.stat().st_size:,} bytes)")
    print(f"    client2: {c2_path.exists()} ({c2_path.stat().st_size:,} bytes)")
    print(f"    client3: {c3_path.exists()} ({c3_path.stat().st_size:,} bytes)")

    df_c1 = pd.read_csv(c1_path)
    df_c2 = pd.read_csv(c2_path)
    df_c3 = pd.read_csv(c3_path)

    print(f"    client1 shape: {df_c1.shape}")
    print(f"    client2 shape: {df_c2.shape}")
    print(f"    client3 shape: {df_c3.shape}")

    # 2. Check feature columns
    with open(ARTIFACTS_DIR / "model_config.json") as f:
        cfg = json.load(f)
    expected_features = cfg["features"]

    print(f"\n[2] Feature columns analysis:")
    print(f"    Expected features count: {len(expected_features)}")
    print(f"    client1 columns count:   {len(df_c1.columns)}")
    print(f"    client2 columns count:   {len(df_c2.columns)}")

    c1_has_all = all(f in df_c1.columns for f in expected_features)
    c2_has_all = all(f in df_c2.columns for f in expected_features)
    print(f"    client1 contains all 45 expected features: {c1_has_all}")
    print(f"    client2 contains all 45 expected features: {c2_has_all}")

    extra_c1 = [c for c in df_c1.columns if c not in expected_features]
    print(f"    client1 non-feature columns: {extra_c1}")

    # 3. Class distributions
    print(f"\n[3] Class distributions:")
    print(f"\n--- client1_traffic.csv (Total: {len(df_c1)}) ---")
    c1_dist = df_c1["label"].value_counts()
    print(f"Unique classes: {len(c1_dist)}")
    for cls, count in c1_dist.items():
        print(f"    {cls:<30} {count:>4} ({count/len(df_c1)*100:>5.1f}%)")

    print(f"\n--- client2_traffic.csv (Total: {len(df_c2)}) ---")
    c2_dist = df_c2["label"].value_counts()
    print(f"Unique classes: {len(c2_dist)}")
    for cls, count in c2_dist.items():
        print(f"    {cls:<30} {count:>4} ({count/len(df_c2)*100:>5.1f}%)")

    # 4. Investigate Source Origin: Matching against P:/project files
    print(f"\n[4] Matching client records against original Colab splits in P:/project:")
    # We will read initial slices from train.csv, validation.csv, test.csv
    # and compare the exact first row of client1 and client2

    c1_first_row = df_c1[expected_features].iloc[0].values
    c2_first_row = df_c2[expected_features].iloc[0].values

    for split_name in ["train.csv", "validation.csv", "test.csv"]:
        split_path = PROJECT_DIR / split_name
        if not split_path.exists():
            print(f"    [SKIP] {split_name} does not exist.")
            continue

        print(f"    Checking {split_name} (reading initial 3000 rows)...")
        df_split = pd.read_csv(split_path, nrows=3000)

        # Check where c1 begins
        # Compare row 0 of c1
        matches_c1 = []
        matches_c2 = []
        for i in range(len(df_split)):
            row = df_split[expected_features].iloc[i].values
            if np.allclose(row.astype(float), c1_first_row.astype(float), rtol=1e-5, atol=1e-5):
                matches_c1.append(i)
            if np.allclose(row.astype(float), c2_first_row.astype(float), rtol=1e-5, atol=1e-5):
                matches_c2.append(i)

        if matches_c1:
            print(f"      -> FOUND client1 row 0 in {split_name} at row index: {matches_c1}")
            # Verify subsequent rows
            start_idx = matches_c1[0]
            slice_match = df_split[expected_features].iloc[start_idx:start_idx+len(df_c1)].values
            is_identical = np.allclose(slice_match.astype(float), df_c1[expected_features].values.astype(float), rtol=1e-5, atol=1e-5)
            print(f"      -> Full client1 slice identical to {split_name}[{start_idx} : {start_idx + len(df_c1)}]: {is_identical}")

        if matches_c2:
            print(f"      -> FOUND client2 row 0 in {split_name} at row index: {matches_c2}")
            start_idx = matches_c2[0]
            slice_match = df_split[expected_features].iloc[start_idx:start_idx+len(df_c2)].values
            is_identical = np.allclose(slice_match.astype(float), df_c2[expected_features].values.astype(float), rtol=1e-5, atol=1e-5)
            print(f"      -> Full client2 slice identical to {split_name}[{start_idx} : {start_idx + len(df_c2)}]: {is_identical}")

    # 5. Overlap between client1 and client2
    print(f"\n[5] Overlap analysis between client partitions:")
    # Check if any row in client1 exists in client2
    c1_tuples = set(tuple(x) for x in df_c1[expected_features].values)
    c2_tuples = set(tuple(x) for x in df_c2[expected_features].values)
    c3_tuples = set(tuple(x) for x in df_c3[expected_features].values)

    c1_c2_overlap = len(c1_tuples.intersection(c2_tuples))
    c1_c3_overlap = len(c1_tuples.intersection(c3_tuples))
    c2_c3_overlap = len(c2_tuples.intersection(c3_tuples))
    print(f"    Overlapping rows between client1 and client2: {c1_c2_overlap} / 600")
    print(f"    Overlapping rows between client1 and client3: {c1_c3_overlap} / 600")
    print(f"    Overlapping rows between client2 and client3: {c2_c3_overlap} / 600")

    # 6. Verify Scaler & Label Encoder on client1 and client2
    print(f"\n[6] Verifying Scaler and Label Encoder on both client partitions:")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    le = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")

    # Client 1
    X_c1 = scaler.transform(df_c1[expected_features].values.astype(np.float32))
    y_c1 = le.transform(df_c1["label"].values)
    print(f"    client1 successfully transformed: X shape={X_c1.shape}, y shape={y_c1.shape}")

    # Client 2
    X_c2 = scaler.transform(df_c2[expected_features].values.astype(np.float32))
    y_c2 = le.transform(df_c2["label"].values)
    print(f"    client2 successfully transformed: X shape={X_c2.shape}, y shape={y_c2.shape}")

    print("\n" + "=" * 70)
    print("PROVENANCE INVESTIGATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
