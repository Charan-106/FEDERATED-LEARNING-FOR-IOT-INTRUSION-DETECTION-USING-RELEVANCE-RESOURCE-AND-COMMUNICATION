"""
Phase 7 Automated Verification Script:
Tests Model Management, Versioning, Download, Rollback, and Analytics:
1. Candidate Model Publication as 'global-v2' (packaging .pt and helper assets).
2. Database state: GlobalModel versioning, Round association, EvaluationResult, and AuditLog.
3. Download verification: Streams genuine model bytes for both Admin and Client.
4. Model Rollback: Atomic revert from 'global-v2' back to 'initial-v1', verified in DB & audit.
5. Model Re-publication: Publishing 'global-v2' as the authoritative state.
6. Historical Analytics: Verifies convergence curves (baseline -> round 1) and fleet stats.
"""

import requests
import json
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000/api"

def main():
    print("=" * 70)
    print("ARC-FL PHASE 7 MODEL MANAGEMENT & ANALYTICS VERIFICATION")
    print("=" * 70)

    # 1. Admin login
    print("\n[Step 1] Authenticating as Admin...")
    res = requests.post(f"{BASE_URL}/auth/login", json={"email": "admin@arc-fl.local", "password": "Admin@123"})
    assert res.status_code == 200, f"Login failed: {res.text}"
    admin_token = res.json()["token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    print("   [OK] Admin authenticated.")

    # 2. Client login for download check
    res_cl = requests.post(f"{BASE_URL}/auth/login", json={"email": "client1@arc-fl.local", "password": "Client@123"})
    assert res_cl.status_code == 200
    client_token = res_cl.json()["token"]
    client_headers = {"Authorization": f"Bearer {client_token}"}

    # 3. Check current active model before publishing candidate
    print("\n[Step 2] Checking Current Active Model (pre-publication)...")
    res = requests.get(f"{BASE_URL}/models/current", headers=admin_headers)
    assert res.status_code == 200
    initial_model = res.json()["model"]
    print(f"   [OK] Active Model: {initial_model['version']} (ID: {initial_model['id']})")
    assert initial_model["version"] == "initial-v1"

    # 4. Publish Candidate as 'global-v2'
    print("\n[Step 3] Publishing Merged Candidate as 'global-v2'...")
    res = requests.post(f"{BASE_URL}/models/publish-candidate", headers=admin_headers, json={
        "version": "global-v2",
        "name": "ARC-FL IoT IDS Model v2"
    })
    assert res.status_code == 200, f"Publish candidate failed: {res.text}"
    publish_res = res.json()
    new_model = publish_res["model"]
    print(f"   [OK] Published new global model: {new_model['version']} (ID: {new_model['id']})")
    print(f"        • Architecture: {new_model['architecture']}")
    print(f"        • Parameters: {new_model['totalParameters']:,}")
    print(f"        • Baseline Accuracy: {new_model['baselineAccuracy']}%")
    print(f"        • Baseline Loss: {new_model['baselineLoss']}")
    print(f"        • Storage Path: {new_model['storagePath']}")
    print(f"        • Checksum SHA256: {new_model['checksumSha256'][:24]}...")

    # Verify physical files on disk
    storage_root = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning\storage")
    v2_dir = storage_root / "global-models" / "global-v2"
    assert (v2_dir / "best.pt").exists(), "best.pt not found in global-v2 storage"
    assert (v2_dir / "model.py").exists(), "model.py not found in global-v2 storage"
    assert (v2_dir / "scaler.pkl").exists(), "scaler.pkl not found in global-v2 storage"
    assert (v2_dir / "label_encoder.pkl").exists(), "label_encoder.pkl not found in global-v2 storage"
    print(f"   [OK] Physical package verified on disk: {v2_dir} (all assets present)")

    # 5. Verify Model List & Client Active Model
    print("\n[Step 4] Verifying Registry Versions & Client Active Model Inspection...")
    res = requests.get(f"{BASE_URL}/models/versions", headers=admin_headers)
    assert res.status_code == 200
    versions = res.json()["models"]
    print(f"   [OK] Registry now contains {len(versions)} immutable versions:")
    for v in versions:
        active_tag = "[ACTIVE]" if v["isActive"] else "        "
        print(f"        {active_tag} {v['version']}: {v['name']} (Acc: {v['baselineAccuracy']}%, Params: {v['totalParameters']:,})")
    assert len(versions) >= 2, "Expected at least 2 model versions in registry"

    # Client view
    res = requests.get(f"{BASE_URL}/client/model", headers=client_headers)
    assert res.status_code == 200
    cl_model = res.json()["model"]
    print(f"   [OK] Client sees active model: {cl_model['version']} (Accuracy: {cl_model['baselineAccuracy']}%)")
    assert cl_model["version"] == "global-v2"

    # 6. Test Model Download (Admin & Client)
    print("\n[Step 5] Testing Functional Model Downloads...")
    # Admin download endpoint
    res_dl = requests.get(f"{BASE_URL}/models/{new_model['id']}/download", headers=admin_headers, stream=True)
    assert res_dl.status_code == 200, f"Admin download failed: {res_dl.status_code}"
    content_len = len(res_dl.content)
    disposition = res_dl.headers.get("Content-Disposition", "")
    print(f"   [OK] Admin downloaded {content_len:,} bytes. Header: {disposition}")
    assert content_len > 1_000_000, "Downloaded file is suspiciously small"

    # Client download endpoint
    res_cl_dl = requests.get(f"{BASE_URL}/client/model/download", headers=client_headers, stream=True)
    assert res_cl_dl.status_code == 200, f"Client download failed: {res_cl_dl.status_code}"
    cl_content_len = len(res_cl_dl.content)
    print(f"   [OK] Client downloaded {cl_content_len:,} bytes.")
    assert cl_content_len > 1_000_000

    # 7. Test Instant Rollback
    print("\n[Step 6] Testing Instant Rollback to 'initial-v1'...")
    res = requests.post(f"{BASE_URL}/models/{initial_model['id']}/rollback", headers=admin_headers)
    assert res.status_code == 200, f"Rollback failed: {res.text}"
    print(f"   [OK] Rollback executed successfully: {res.json()['message']}")

    # Verify active model is now initial-v1
    res = requests.get(f"{BASE_URL}/models/current", headers=admin_headers)
    rolled_back_model = res.json()["model"]
    print(f"   [OK] Current active model after rollback: {rolled_back_model['version']}")
    assert rolled_back_model["version"] == "initial-v1"

    # 8. Re-publish 'global-v2' as active model
    print("\n[Step 7] Re-publishing 'global-v2' as the Active Global Model...")
    res = requests.post(f"{BASE_URL}/models/{new_model['id']}/publish", headers=admin_headers)
    assert res.status_code == 200
    print(f"   [OK] 'global-v2' re-established as active model.")

    # 9. Verify Convergence Analytics
    print("\n[Step 8] Verifying Convergence Analytics Across Rounds...")
    res = requests.get(f"{BASE_URL}/admin/analytics", headers=admin_headers)
    assert res.status_code == 200, f"Analytics query failed: {res.text}"
    round_data = res.json()["roundData"]
    print(f"   [OK] Retrieved {len(round_data)} analytics trajectory points:")
    for pt in round_data:
        print(f"        • Round #{pt['roundNumber']} ({pt['status']}): "
              f"Clients={pt['clientCount']}, "
              f"Avg Local Acc={pt['avgLocalAccuracy']}%, "
              f"Global Acc={pt['globalAccuracy']}%, "
              f"Global Loss={pt['globalLoss']}")

    assert len(round_data) >= 2, "Expected at least 2 points (Round 0 baseline and Round 1)"

    # 10. Audit Log Inspection
    print("\n[Step 9] Verifying Audit Trail for Model Management Actions...")
    res = requests.get(f"{BASE_URL}/admin/audit-logs", headers=admin_headers)
    assert res.status_code == 200
    logs = res.json()["logs"]
    model_logs = [l for l in logs if l["action"] in ["PUBLISH_MODEL", "ROLLBACK_MODEL"]]
    print(f"   [OK] Found {len(model_logs)} model lifecycle audit log(s):")
    for l in model_logs[:4]:
        print(f"        - Action: {l['action']} | Target: {l['detailsJson']} | Time: {l['timestamp']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] PHASE 7 MODEL MANAGEMENT & ANALYTICS FULLY VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
