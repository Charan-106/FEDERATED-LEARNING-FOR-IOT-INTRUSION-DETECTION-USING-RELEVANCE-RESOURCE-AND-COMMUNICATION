"""
Unit Test Suite for Experiment 004 Scoring and Training Improvements
Verifies:
1. Deterministic seeding in train_local
2. Feature noise outlier detection on continuous data
3. Out-of-fold reference label consistency scoring on clean vs corrupted datasets
4. Multi-factor quality aggregation in ARCScoringEngine
5. ARCScoringEngine coefficient configurations (0.4/0.3/0.3 vs 0.6/0.2/0.2)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning")
sys.path.insert(0, str(WORKSPACE_ROOT / "ml-service"))

from app.arc.scoring import ARCScoringEngine
from app.training.trainer import train_local

BASE_MODEL_PATH = WORKSPACE_ROOT / "storage" / "global-models" / "initial-v1" / "best.pt"
CLEAN_DATASET = WORKSPACE_ROOT / "storage" / "experiments" / "exp_004" / "datasets" / "client1_clean_6000.csv"
CORRUPT_DATASET = WORKSPACE_ROOT / "storage" / "experiments" / "exp_004" / "datasets" / "client3_corrupted_1000.csv"


class TestExperiment004Improvements(unittest.TestCase):

    def test_seed_reproducibility(self):
        """Verify identical seeds produce identical local training losses and update norms."""
        res1 = train_local(
            dataset_path=str(CLEAN_DATASET),
            global_model_path=str(BASE_MODEL_PATH),
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            max_samples=200,
            seed=42,
        )
        res2 = train_local(
            dataset_path=str(CLEAN_DATASET),
            global_model_path=str(BASE_MODEL_PATH),
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            max_samples=200,
            seed=42,
        )
        self.assertEqual(res1["loss"], res2["loss"])
        self.assertEqual(res1["update_norm"], res2["update_norm"])

    def test_label_noise_detection_contrast(self):
        """Verify that label consistency score drastically separates clean from corrupted labels."""
        res_clean = train_local(
            dataset_path=str(CLEAN_DATASET),
            global_model_path=str(BASE_MODEL_PATH),
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            max_samples=200,
            seed=42,
        )
        res_corrupt = train_local(
            dataset_path=str(CORRUPT_DATASET),
            global_model_path=str(BASE_MODEL_PATH),
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            max_samples=200,
            seed=42,
        )
        clean_label_score = res_clean["quality_metrics"]["label_agreement_score"]
        corrupt_label_score = res_corrupt["quality_metrics"]["label_agreement_score"]

        # Clean data should have > 90% label agreement
        self.assertGreater(clean_label_score, 90.0)
        # Corrupted data should have significantly lower agreement
        self.assertLess(corrupt_label_score, 50.0)
        print(f"\n[Test] Clean label agreement: {clean_label_score:.2f}% vs Corrupted: {corrupt_label_score:.2f}%")

    def test_feature_noise_detection(self):
        """Verify that feature noise score identifies outlier violations."""
        res_clean = train_local(
            dataset_path=str(CLEAN_DATASET),
            global_model_path=str(BASE_MODEL_PATH),
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            max_samples=200,
            seed=42,
        )
        res_corrupt = train_local(
            dataset_path=str(CORRUPT_DATASET),
            global_model_path=str(BASE_MODEL_PATH),
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            max_samples=200,
            seed=42,
        )
        clean_noise_score = res_clean["quality_metrics"]["feature_noise_score"]
        corrupt_noise_score = res_corrupt["quality_metrics"]["feature_noise_score"]

        self.assertGreater(clean_noise_score, 95.0)
        self.assertLess(corrupt_noise_score, 80.0)
        print(f"[Test] Clean feature noise score: {clean_noise_score:.2f}% vs Corrupted: {corrupt_noise_score:.2f}%")

    def test_arc_coefficient_comparison(self):
        """Verify that ARCScoringEngine supports both original (0.4/0.3/0.3) and relevance-focused (0.6/0.2/0.2)."""
        engine_orig = ARCScoringEngine({"lambda_relevance": 0.40, "lambda_resource": 0.30, "lambda_communication": 0.30})
        engine_rel = ARCScoringEngine({"lambda_relevance": 0.60, "lambda_resource": 0.20, "lambda_communication": 0.20})

        # Test with high relevance (R=90) but low hardware (S=20, C=30)
        score_orig = engine_orig.compute_overall(90.0, 20.0, 30.0)
        score_rel = engine_rel.compute_overall(90.0, 20.0, 30.0)

        # Expected:
        # Orig: 0.4(90) + 0.3(20) + 0.3(30) = 36 + 6 + 9 = 51.0
        # Rel:  0.6(90) + 0.2(20) + 0.2(30) = 54 + 4 + 6 = 64.0
        self.assertAlmostEqual(score_orig, 51.0, places=2)
        self.assertAlmostEqual(score_rel, 64.0, places=2)
        print(f"[Test] Score with R=90, S=20, C=30 -> Orig: {score_orig:.2f} vs Relevance-Focused: {score_rel:.2f}")


if __name__ == "__main__":
    unittest.main()
