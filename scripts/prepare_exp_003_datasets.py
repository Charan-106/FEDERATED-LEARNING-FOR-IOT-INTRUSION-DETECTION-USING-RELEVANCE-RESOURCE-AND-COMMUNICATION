"""
Prepare Datasets for Experiment 003: Heterogeneous Client Evaluation

Prepares verified, disjoint datasets for:
- Scenario A: Unequal Data Volume (6000, 3000, 1000 samples from P:/project/train.csv)
- Scenario B: Non-IID Class Distribution (Attack Family Specialization from P:/project/train.csv)
- Scenario C: Uses Scenario B's balanced clean datasets with simulated resource/comm telemetry
- Scenario D: Controlled Data Quality Heterogeneity (Clean C1, C2 + Corrupted C3)

Strict constraints:
- Zero data duplication
- Preserves class labels and feature ordering
- Leaves P:/project/test.csv strictly quarantined
"""

import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp_003_prep")

WORKSPACE_ROOT = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning")
EXP_003_DIR = WORKSPACE_ROOT / "storage" / "experiments" / "exp_003"
TRAIN_CSV = Path(r"P:\project\train.csv")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def prepare_scenario_a():
    logger.info("--- Preparing Scenario A: Unequal Data Volume ---")
    scen_dir = EXP_003_DIR / "scenario_a"
    scen_dir.mkdir(parents=True, exist_ok=True)

    # Read verified 10,000 samples from train.csv
    # Rows 0 to 6000 -> Client 1 (6000 samples)
    # Rows 6000 to 9000 -> Client 2 (3000 samples)
    # Rows 9000 to 10000 -> Client 3 (1000 samples)
    df_10k = pd.read_csv(TRAIN_CSV, nrows=10000)

    c1_df = df_10k.iloc[0:6000].copy()
    c2_df = df_10k.iloc[6000:9000].copy()
    c3_df = df_10k.iloc[9000:10000].copy()

    c1_path = scen_dir / "client1_vol6000.csv"
    c2_path = scen_dir / "client2_vol3000.csv"
    c3_path = scen_dir / "client3_vol1000.csv"

    c1_df.to_csv(c1_path, index=False)
    c2_df.to_csv(c2_path, index=False)
    c3_df.to_csv(c3_path, index=False)

    logger.info(f"Scenario A saved: C1={len(c1_df)} ({sha256_file(c1_path)[:16]}), C2={len(c2_df)} ({sha256_file(c2_path)[:16]}), C3={len(c3_df)} ({sha256_file(c3_path)[:16]})")


def prepare_scenario_b():
    logger.info("--- Preparing Scenario B: Non-IID Class Distribution ---")
    scen_dir = EXP_003_DIR / "scenario_b"
    scen_dir.mkdir(parents=True, exist_ok=True)

    # Read pool of rows 10,000 to 50,000 to ensure completely disjoint from Scenario A
    df_pool = pd.read_csv(TRAIN_CSV, skiprows=range(1, 10001), nrows=40000)
    logger.info(f"Read {len(df_pool)} rows from train.csv (pool for Non-IID selection)")

    # Define attack class family sets:
    # Client 1: Volumetric Floods (ICMP, UDP, TCP Floods) + Benign
    c1_classes = [
        "DDoS-ICMP_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood",
        "DoS-UDP_Flood", "DoS-TCP_Flood", "BenignTraffic"
    ]
    # Client 2: Botnet & Network Protocol Exploits (Mirai, SYN, PSHACK, RSTFIN) + Benign
    c2_classes = [
        "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain",
        "DDoS-SYN_Flood", "DDoS-PSHACK_Flood", "DDoS-RSTFINFlood",
        "DDoS-SynonymousIP_Flood", "DoS-SYN_Flood", "BenignTraffic"
    ]
    # Client 3: Reconnaissance, Spoofing, Fragmentation & Web/Injection Attacks + Benign
    c3_classes = [
        "Recon-HostDiscovery", "Recon-PortScan", "Recon-OSScan", "Recon-PingSweep",
        "MITM-ArpSpoofing", "DNS_Spoofing", "DDoS-ACK_Fragmentation",
        "DDoS-ICMP_Fragmentation", "DDoS-UDP_Fragmentation", "VulnerabilityScan",
        "DoS-HTTP_Flood", "DDoS-HTTP_Flood", "DDoS-SlowLoris", "DictionaryBruteForce",
        "SqlInjection", "Backdoor_Malware", "CommandInjection", "BrowserHijacking",
        "Uploading_Attack", "BenignTraffic"
    ]

    used_indices = set()

    # Client 1 selection (1000 samples)
    c1_cand = df_pool[df_pool["label"].isin(c1_classes) & ~df_pool.index.isin(used_indices)]
    c1_df = c1_cand.head(1000).copy()
    used_indices.update(c1_df.index)

    # Client 2 selection (1000 samples)
    c2_cand = df_pool[df_pool["label"].isin(c2_classes) & ~df_pool.index.isin(used_indices)]
    c2_df = c2_cand.head(1000).copy()
    used_indices.update(c2_df.index)

    # Client 3 selection (1000 samples)
    # Collect as many rare / specialized attacks as possible, then fill remaining up to 1000
    c3_specialized = df_pool[df_pool["label"].isin(c3_classes) & ~df_pool.index.isin(used_indices)]
    if len(c3_specialized) >= 1000:
        c3_df = c3_specialized.head(1000).copy()
    else:
        # Fill remainder with other unused rows to guarantee exactly 1000 samples
        needed = 1000 - len(c3_specialized)
        remainder = df_pool[~df_pool.index.isin(used_indices) & ~df_pool.index.isin(c3_specialized.index)].head(needed)
        c3_df = pd.concat([c3_specialized, remainder], ignore_index=True)
    used_indices.update(c3_df.index)

    # Verify disjointness
    assert len(set(c1_df.index).intersection(set(c2_df.index))) == 0
    assert len(set(c1_df.index).intersection(set(c3_df.index))) == 0
    assert len(set(c2_df.index).intersection(set(c3_df.index))) == 0

    c1_path = scen_dir / "client1_non_iid.csv"
    c2_path = scen_dir / "client2_non_iid.csv"
    c3_path = scen_dir / "client3_non_iid.csv"

    c1_df.to_csv(c1_path, index=False)
    c2_df.to_csv(c2_path, index=False)
    c3_df.to_csv(c3_path, index=False)

    logger.info(f"Scenario B saved: C1={len(c1_df)} ({len(c1_df['label'].unique())} classes), C2={len(c2_df)} ({len(c2_df['label'].unique())} classes), C3={len(c3_df)} ({len(c3_df['label'].unique())} classes)")


def prepare_scenario_d():
    logger.info("--- Preparing Scenario D: Data Quality Heterogeneity ---")
    scen_dir = EXP_003_DIR / "scenario_d"
    scen_dir.mkdir(parents=True, exist_ok=True)

    # Read clean 1000-sample slices from pool for C1 and C2 (rows 50000 to 52000)
    df_pool = pd.read_csv(TRAIN_CSV, skiprows=range(1, 50001), nrows=3000)

    c1_clean = df_pool.iloc[0:1000].copy()
    c2_clean = df_pool.iloc[1000:2000].copy()
    c3_clean = df_pool.iloc[2000:3000].copy()

    # Save clean control versions
    c1_clean.to_csv(scen_dir / "client1_clean.csv", index=False)
    c2_clean.to_csv(scen_dir / "client2_clean.csv", index=False)
    c3_clean.to_csv(scen_dir / "client3_clean_control.csv", index=False)

    # Apply controlled corruption to Client 3
    # 1. 30% missing values (NaN) across feature columns
    # 2. Gaussian noise added to numerical values
    # 3. 35% label noise (random swaps among known classes)
    np.random.seed(42)
    c3_corrupted = c3_clean.copy()

    feature_cols = [c for c in c3_corrupted.columns if c not in ("label", "Telnet")]

    # Feature corruption: introduce NaNs in 30% of cells
    mask = np.random.rand(*c3_corrupted[feature_cols].shape) < 0.30
    c3_corrupted[feature_cols] = c3_corrupted[feature_cols].mask(mask, np.nan)

    # Additional Gaussian noise on 20% of remaining numerical cells
    noise_mask = np.random.rand(*c3_corrupted[feature_cols].shape) < 0.20
    noise = np.random.normal(0, 5.0, size=c3_corrupted[feature_cols].shape)
    c3_corrupted[feature_cols] = c3_corrupted[feature_cols] + (noise_mask * noise)

    # Label corruption: flip 35% of labels to random classes
    unique_labels = c3_clean["label"].unique().tolist()
    flip_indices = c3_corrupted.sample(frac=0.35, random_state=42).index
    random_labels = np.random.choice(unique_labels, size=len(flip_indices))
    c3_corrupted.loc[flip_indices, "label"] = random_labels

    c3_corr_path = scen_dir / "client3_corrupted.csv"
    c3_corrupted.to_csv(c3_corr_path, index=False)

    missing_ratio = float(c3_corrupted.isna().sum().sum()) / (c3_corrupted.shape[0] * c3_corrupted.shape[1])
    logger.info(f"Scenario D saved: C1=Clean, C2=Clean, C3=Corrupted (Missing ratio={missing_ratio:.4f}, Label noise=35%)")


def main():
    EXP_003_DIR.mkdir(parents=True, exist_ok=True)
    prepare_scenario_a()
    prepare_scenario_b()
    prepare_scenario_d()
    logger.info("All Experiment 003 datasets successfully prepared!")


if __name__ == "__main__":
    main()
