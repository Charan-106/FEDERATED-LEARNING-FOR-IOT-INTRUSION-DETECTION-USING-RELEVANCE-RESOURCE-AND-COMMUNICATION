"""
ARC-FL Benchmark Dataset Generator
Generates fixed, deterministic evaluation benchmarks from P:\project\validation.csv
in accordance with RESEARCH_EVALUATION_PROTOCOL.md.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Original Colab class distribution proportions (from ciciot2023_fl_base_model_v3.ipynb Cell 20)
COLAB_CLASS_SUPPORT = {
    'Backdoor_Malware': 483,
    'BenignTraffic': 44999,
    'BrowserHijacking': 879,
    'CommandInjection': 811,
    'DDoS-ACK_Fragmentation': 41240,
    'DDoS-HTTP_Flood': 4316,
    'DDoS-ICMP_Flood': 29567,
    'DDoS-ICMP_Fragmentation': 44370,
    'DDoS-PSHACK_Flood': 32174,
    'DDoS-RSTFINFlood': 26708,
    'DDoS-SYN_Flood': 36525,
    'DDoS-SlowLoris': 3514,
    'DDoS-SynonymousIP_Flood': 41789,
    'DDoS-TCP_Flood': 33150,
    'DDoS-UDP_Flood': 45000,
    'DDoS-UDP_Fragmentation': 43039,
    'DNS_Spoofing': 26831,
    'DictionaryBruteForce': 1960,
    'DoS-HTTP_Flood': 10768,
    'DoS-SYN_Flood': 40930,
    'DoS-TCP_Flood': 37648,
    'DoS-UDP_Flood': 43462,
    'MITM-ArpSpoofing': 45000,
    'Mirai-greeth_flood': 35260,
    'Mirai-greip_flood': 36212,
    'Mirai-udpplain': 45000,
    'Recon-HostDiscovery': 20152,
    'Recon-OSScan': 14717,
    'Recon-PingSweep': 339,
    'Recon-PortScan': 12319,
    'SqlInjection': 787,
    'Uploading_Attack': 188,
    'VulnerabilityScan': 5607,
    'XSS': 577,
}

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    workspace_root = Path(__file__).resolve().parent.parent
    benchmarks_dir = workspace_root / "storage" / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    results_dir = benchmarks_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    val_path = Path(r"P:\project\validation.csv")
    test_path = Path(r"P:\project\test.csv")

    if not val_path.exists():
        print(f"ERROR: Validation dataset not found at {val_path}")
        sys.exit(1)

    print(f"Loading {val_path} ...")
    df_val = pd.read_csv(val_path)
    print(f"Loaded {len(df_val):,} rows.")

    # -------------------------------------------------------------
    # 1. Generate Benchmark A: Primary Capped (Colab Distribution)
    # -------------------------------------------------------------
    print("\n--- Generating Benchmark A: Primary Capped (50,001 rows) ---")
    total_colab = sum(COLAB_CLASS_SUPPORT.values())
    colab_weights = {k: v / total_colab for k, v in COLAB_CLASS_SUPPORT.items()}
    target_primary_total = 50000

    sampled_dfs = []
    rng_primary = np.random.default_rng(42)

    for cls, weight in colab_weights.items():
        n_samples = int(round(weight * target_primary_total))
        cls_df = df_val[df_val['label'] == cls]
        available = len(cls_df)
        n_take = min(n_samples, available)
        sampled = cls_df.sample(n=n_take, random_state=42)
        sampled_dfs.append(sampled)

    df_primary = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    primary_path = benchmarks_dir / "benchmark_primary_capped.csv"
    df_primary.to_csv(primary_path, index=False)
    primary_hash = sha256_file(primary_path)
    print(f"Saved: {primary_path} ({len(df_primary):,} rows)")
    print(f"SHA-256: {primary_hash}")

    # -------------------------------------------------------------
    # 2. Generate Benchmark B: Secondary Natural (Natural Distribution)
    # -------------------------------------------------------------
    print("\n--- Generating Benchmark B: Secondary Natural (50,000 rows) ---")
    df_secondary = df_val.sample(n=50000, random_state=42).reset_index(drop=True)
    secondary_path = benchmarks_dir / "benchmark_secondary_natural.csv"
    df_secondary.to_csv(secondary_path, index=False)
    secondary_hash = sha256_file(secondary_path)
    print(f"Saved: {secondary_path} ({len(df_secondary):,} rows)")
    print(f"SHA-256: {secondary_hash}")

    # -------------------------------------------------------------
    # 3. Checkpoint Quarantined Test Dataset
    # -------------------------------------------------------------
    print("\n--- Auditing Quarantined Benchmark C (Test Dataset) ---")
    test_hash = sha256_file(test_path) if test_path.exists() else "NOT_FOUND"
    print(f"Path: {test_path}")
    print(f"SHA-256: {test_hash}")
    print("Status: STRICTLY QUARANTINED (Forbidden from training, ARC, or tuning)")

    # -------------------------------------------------------------
    # 4. Save Manifest
    # -------------------------------------------------------------
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "1.0",
        "benchmarks": {
            "primary_capped": {
                "file": "benchmark_primary_capped.csv",
                "path": str(primary_path),
                "sha256": primary_hash,
                "rows": len(df_primary),
                "classes": int(df_primary['label'].nunique()),
                "class_distribution": df_primary['label'].value_counts().to_dict(),
                "description": "Stratified sample matching original Google Colab class-capped distribution"
            },
            "secondary_natural": {
                "file": "benchmark_secondary_natural.csv",
                "path": str(secondary_path),
                "sha256": secondary_hash,
                "rows": len(df_secondary),
                "classes": int(df_secondary['label'].nunique()),
                "class_distribution": df_secondary['label'].value_counts().to_dict(),
                "description": "Natural traffic distribution sample from P:\\project\\validation.csv"
            },
            "quarantined_test": {
                "file": "test.csv",
                "path": str(test_path),
                "sha256": test_hash,
                "rows": 1176851,
                "status": "QUARANTINED"
            }
        }
    }

    manifest_path = benchmarks_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print("Benchmark generation completed successfully.")

if __name__ == "__main__":
    main()
