import sys
import hashlib
from pathlib import Path
import torch

WORKSPACE = Path("C:/Users/chara/.gemini/antigravity/scratch/arc-federated-learning")
STORAGE = WORKSPACE / "storage"

print("=" * 80)
print("READ-ONLY LIVE DEMO VERIFICATION: FILES & TENSORS")
print("=" * 80)

# 1. Base Model initial-v1
init_path = STORAGE / "global-models" / "initial-v1" / "best.pt"
assert init_path.exists(), "initial-v1 best.pt missing"
init_bytes = init_path.read_bytes()
init_hash = hashlib.sha256(init_bytes).hexdigest()
expected_base_hash = "57b7afd9d01c30b48101fc2660d699ea1b5790104c00e534e00e090b517d4993"
print(f"1. Base Model initial-v1:")
print(f"   Path: {init_path}")
print(f"   Size: {len(init_bytes):,} bytes")
print(f"   SHA-256: {init_hash}")
print(f"   Matches expected base hash: {init_hash == expected_base_hash}")
assert init_hash == expected_base_hash

# 2. Candidate Model Checkpoint
cand_path = STORAGE / "candidates" / "d6858f26-dd65-484d-b393-87981ae8fa95.pt"
assert cand_path.exists(), "Candidate checkpoint missing"
cand_bytes = cand_path.read_bytes()
cand_hash = hashlib.sha256(cand_bytes).hexdigest()
recorded_cand_hash = "b0d95994d75be435f6c565e156b371207b0ba8ed126eb5ba983832da525d0e8b"
print(f"\n2. Candidate Model Checkpoint:")
print(f"   Path: {cand_path}")
print(f"   Size: {len(cand_bytes):,} bytes")
print(f"   SHA-256: {cand_hash}")
print(f"   Matches recorded hash: {cand_hash == recorded_cand_hash}")
assert cand_hash == recorded_cand_hash

# 3. Published Model global-v22
v22_path = STORAGE / "global-models" / "global-v22" / "best.pt"
assert v22_path.exists(), "global-v22 best.pt missing"
v22_bytes = v22_path.read_bytes()
v22_hash = hashlib.sha256(v22_bytes).hexdigest()
print(f"\n3. Published Model global-v22:")
print(f"   Path: {v22_path}")
print(f"   Size: {len(v22_bytes):,} bytes")
print(f"   SHA-256: {v22_hash}")
print(f"   Matches candidate checkpoint: {v22_hash == cand_hash}")
assert v22_hash == cand_hash

# 4. Client 1 Update File
c1_update = STORAGE / "client-updates" / "4b565adb-02d6-47ab-93c9-2b69b25e99d3.pt"
assert c1_update.exists(), "Client 1 update missing"
c1_bytes = c1_update.read_bytes()
c1_hash = hashlib.sha256(c1_bytes).hexdigest()
recorded_c1_hash = "c95b892847bb55fcbd12fc2dd31ad25201fa6d6363f5f191ab8d9d917323f56c"
print(f"\n4. Client 1 Update Checkpoint:")
print(f"   Path: {c1_update}")
print(f"   Size: {len(c1_bytes):,} bytes")
print(f"   SHA-256: {c1_hash}")
print(f"   Matches recorded submission hash: {c1_hash == recorded_c1_hash}")
assert c1_hash == recorded_c1_hash

# 5. Client 2 Update File
c2_update = STORAGE / "client-updates" / "ee3f946a-1dcb-4b28-9480-44007d803b92.pt"
assert c2_update.exists(), "Client 2 update missing"
c2_bytes = c2_update.read_bytes()
c2_hash = hashlib.sha256(c2_bytes).hexdigest()
recorded_c2_hash = "e0bcbbf54063fe88b1b31a4eada92c5c2ab029e17236eab9ae872d91777595dc"
print(f"\n5. Client 2 Update Checkpoint:")
print(f"   Path: {c2_update}")
print(f"   Size: {len(c2_bytes):,} bytes")
print(f"   SHA-256: {c2_hash}")
print(f"   Matches recorded submission hash: {c2_hash == recorded_c2_hash}")
assert c2_hash == recorded_c2_hash

# 6. Tensor Deltas and State Dicts
print("\n6. PyTorch Tensor State Dict Audit:")
init_sd = torch.load(init_path, map_location="cpu")["model"]
cand_sd = torch.load(cand_path, map_location="cpu")["model"]
v22_sd = torch.load(v22_path, map_location="cpu")["model"]
c1_sd = torch.load(c1_update, map_location="cpu")["model"]
c2_sd = torch.load(c2_update, map_location="cpu")["model"]

print(f"   Total layer tensor count: {len(init_sd)}")
assert len(init_sd) == len(cand_sd) == len(v22_sd) == len(c1_sd) == len(c2_sd) == 82
print("   All models contain exactly 82 tensors.")

# Calculate parameter differences
diff_c1 = torch.sqrt(sum((c1_sd[k] - init_sd[k]).pow(2).sum() for k in init_sd if init_sd[k].is_floating_point())).item()
diff_c2 = torch.sqrt(sum((c2_sd[k] - init_sd[k]).pow(2).sum() for k in init_sd if init_sd[k].is_floating_point())).item()
diff_cand = torch.sqrt(sum((cand_sd[k] - init_sd[k]).pow(2).sum() for k in init_sd if init_sd[k].is_floating_point())).item()
diff_v22 = torch.sqrt(sum((v22_sd[k] - init_sd[k]).pow(2).sum() for k in init_sd if init_sd[k].is_floating_point())).item()

print(f"   ||Δθ_1||_2 (Client 1 vs Base): {diff_c1:.4f}")
print(f"   ||Δθ_2||_2 (Client 2 vs Base): {diff_c2:.4f}")
print(f"   ||Δθ_cand||_2 (Candidate vs Base): {diff_cand:.4f}")
print(f"   ||Δθ_v22||_2 (global-v22 vs Base): {diff_v22:.4f}")
assert abs(diff_cand - diff_v22) < 1e-6, "Candidate and global-v22 tensors must be identical"

# Check weighted average formula:
# candidate delta = eta_g * (w1 * delta_c1 + w2 * delta_c2) with eta_g = 0.5
w1 = 0.4903526245958926
w2 = 0.5096473754041074
eta_g = 0.5

max_diff = 0.0
for k in init_sd:
    if init_sd[k].is_floating_point():
        expected_k = init_sd[k] + eta_g * (w1 * (c1_sd[k] - init_sd[k]) + w2 * (c2_sd[k] - init_sd[k]))
        actual_k = cand_sd[k]
        diff = torch.max(torch.abs(expected_k - actual_k)).item()
        if diff > max_diff:
            max_diff = diff

print(f"   Max deviation between mathematical aggregation formula and candidate tensor: {max_diff:.2e}")
assert max_diff < 1e-4, f"Mathematical aggregation discrepancy too large: {max_diff}"
print("   PERFECT MATHEMATICAL MATCH: candidate model is exact weighted average with eta_g = 0.5!")
print("=" * 80)
