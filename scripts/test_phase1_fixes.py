"""
Unit Tests for Phase 1 & Phase 2 Corrections
Tests:
1. BatchNorm running statistics freeze in local training
2. Affine parameters (weight, bias) trainability
3. FedProx proximal penalty calculation
4. num_batches_tracked aggregation without lifetime sum
5. Correct dtypes for floating point and integer buffers
"""

import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ML_SERVICE_DIR = WORKSPACE_ROOT / "ml-service"
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(ML_SERVICE_DIR))

from app.models.resmlp import build_model, load_checkpoint
from app.training.trainer import train_local
from app.aggregation.fedavg import aggregate


class TestPhase1AndPhase2Fixes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_path = WORKSPACE_ROOT / "storage" / "global-models" / "initial-v1" / "best.pt"
        cls.dataset_path = WORKSPACE_ROOT / "storage" / "datasets" / "client1_traffic.csv"
        cls.base_ckpt = torch.load(cls.base_path, map_location="cpu", weights_only=False)
        cls.base_state = cls.base_ckpt["model"]

    def test_01_freeze_bn_running_stats(self):
        """Verify that freeze_bn_stats=True preserves running_mean and running_var identically."""
        res = train_local(
            dataset_path=str(self.dataset_path),
            global_model_path=str(self.base_path),
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            freeze_bn_stats=True,
            mu=0.0,
        )

        up_ckpt = torch.load(res["update_path"], map_location="cpu", weights_only=False)
        up_state = up_ckpt["model"]

        # Check in_bn running_mean and running_var
        diff_mean = torch.max(torch.abs(up_state["in_bn.running_mean"] - self.base_state["in_bn.running_mean"])).item()
        diff_var = torch.max(torch.abs(up_state["in_bn.running_var"] - self.base_state["in_bn.running_var"])).item()
        diff_tracked = (up_state["in_bn.num_batches_tracked"] - self.base_state["in_bn.num_batches_tracked"]).item()

        self.assertEqual(diff_mean, 0.0, "BatchNorm running_mean drifted despite freeze_bn_stats=True")
        self.assertEqual(diff_var, 0.0, "BatchNorm running_var drifted despite freeze_bn_stats=True")
        self.assertEqual(diff_tracked, 0, "BatchNorm num_batches_tracked changed despite freeze_bn_stats=True")

        # Check residual blocks BN running_mean and running_var
        for k in self.base_state.keys():
            if "running_mean" in k or "running_var" in k:
                d = torch.max(torch.abs(up_state[k] - self.base_state[k])).item()
                self.assertEqual(d, 0.0, f"BN buffer {k} drifted despite freeze_bn_stats=True")

    def test_02_bn_affine_trainability(self):
        """Verify that BatchNorm affine weights and biases still receive gradients."""
        model = build_model(45, 34)
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        # Dummy forward pass and backward pass
        x = torch.randn(8, 45)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Verify that in_bn.weight and bias have gradients
        self.assertIsNotNone(model.in_bn.weight.grad)
        self.assertIsNotNone(model.in_bn.bias.grad)
        self.assertGreater(torch.norm(model.in_bn.weight.grad).item(), 0.0)

    def test_03_fedprox_regularization(self):
        """Verify that FedProx mu > 0 penalizes drift from the global model."""
        res_no_prox = train_local(
            dataset_path=str(self.dataset_path),
            global_model_path=str(self.base_path),
            epochs=1,
            batch_size=32,
            learning_rate=0.001,
            freeze_bn_stats=True,
            mu=0.0,
        )
        res_prox = train_local(
            dataset_path=str(self.dataset_path),
            global_model_path=str(self.base_path),
            epochs=1,
            batch_size=32,
            learning_rate=0.001,
            freeze_bn_stats=True,
            mu=0.1,
        )

        up_no_prox = torch.load(res_no_prox["update_path"], map_location="cpu", weights_only=False)["model"]
        up_prox = torch.load(res_prox["update_path"], map_location="cpu", weights_only=False)["model"]

        drift_no_prox = torch.norm(up_no_prox["head.5.weight"] - self.base_state["head.5.weight"]).item()
        drift_prox = torch.norm(up_prox["head.5.weight"] - self.base_state["head.5.weight"]).item()

        # FedProx with mu=0.1 must constrain the parameter drift
        self.assertLess(drift_prox, drift_no_prox, "FedProx failed to constrain parameter drift")

    def test_04_num_batches_tracked_aggregation(self):
        """Verify that num_batches_tracked does NOT sum lifetime counters across clients."""
        base_tracked = self.base_state["in_bn.num_batches_tracked"].item()

        # Create two simulated client states
        s1 = {k: v.clone() for k, v in self.base_state.items()}
        s2 = {k: v.clone() for k, v in self.base_state.items()}
        s1["in_bn.num_batches_tracked"] = torch.tensor(base_tracked + 19, dtype=torch.int64)
        s2["in_bn.num_batches_tracked"] = torch.tensor(base_tracked + 19, dtype=torch.int64)

        p1 = WORKSPACE_ROOT / "storage" / "client-updates" / "test_s1.pt"
        p2 = WORKSPACE_ROOT / "storage" / "client-updates" / "test_s2.pt"
        torch.save({"model": s1, "update_id": "test_s1"}, str(p1))
        torch.save({"model": s2, "update_id": "test_s2"}, str(p2))

        submissions = [
            {"client_id": "c1", "submission_id": "s1", "model_path": str(p1), "normalized_weight": 0.5},
            {"client_id": "c2", "submission_id": "s2", "model_path": str(p2), "normalized_weight": 0.5},
        ]

        res = aggregate(submissions, base_model_path=str(self.base_path), server_learning_rate=0.5)
        cand_ckpt = torch.load(res["candidate_path"], map_location="cpu", weights_only=False)["model"]

        cand_tracked = cand_ckpt["in_bn.num_batches_tracked"].item()
        # Must NOT be 2 * base_tracked + 38!
        self.assertNotEqual(cand_tracked, base_tracked * 2 + 38, "Aggregator incorrectly summed lifetime batch counters!")
        self.assertEqual(cand_tracked, base_tracked + 19, f"Expected {base_tracked + 19}, got {cand_tracked}")
        self.assertEqual(cand_ckpt["in_bn.num_batches_tracked"].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
