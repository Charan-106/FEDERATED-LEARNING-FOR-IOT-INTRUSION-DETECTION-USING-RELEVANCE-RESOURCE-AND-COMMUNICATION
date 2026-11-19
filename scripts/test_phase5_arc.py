"""
Phase 5 Automated Verification Script:
Validates the complete ARC Scoring Engine & Dynamic Configuration System:
1. Mathematical Bounds & Edge Cases Verification (FastAPI ML Service direct calculation)
2. Admin Dynamic Weight Configuration & Validation (Reject non-1.0 sum, accept valid weights)
3. Audit Trail Verification (Checking AuditLog table via API)
4. Formula Versioning & History Tracking
5. Live Recalculation of Fleet Submissions
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/api"
ML_URL = "http://127.0.0.1:8000"

def main():
    print("=" * 70)
    print("ARC-FL PHASE 5 SCORING & DYNAMIC CONFIGURATION VERIFICATION")
    print("=" * 70)

    # 1. Admin login
    print("\n[Step 1] Admin Authentication...")
    res = requests.post(f"{BASE_URL}/auth/login", json={"email": "admin@arc-fl.local", "password": "Admin@123"})
    assert res.status_code == 200, f"Admin login failed: {res.text}"
    token = res.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   [OK] Admin authenticated.")

    # 2. Test ML Service ARC Edge Cases Directly
    print("\n[Step 2] Testing ARC Mathematical Bounds & Edge Cases...")
    ml_headers = {"X-API-Key": "arc-ml-secret-api-key-2026"}
    
    # Edge Case A: Usable low-latency vs extreme high-latency penalty
    res_normal = requests.post(f"{ML_URL}/arc/calculate", headers=ml_headers, json={
        "submission_id": "test-edge-1",
        "client_id": "client-test",
        "round_id": "round-test",
        "comm_metrics": {"bandwidth_mbps": 50.0, "latency_ms": 20.0, "reliability_pct": 100.0}
    })
    res_high_lat = requests.post(f"{ML_URL}/arc/calculate", headers=ml_headers, json={
        "submission_id": "test-edge-2",
        "client_id": "client-test",
        "round_id": "round-test",
        "comm_metrics": {"bandwidth_mbps": 50.0, "latency_ms": 600.0, "reliability_pct": 100.0}
    })
    lat_normal = res_normal.json()["latency_score"]
    lat_penalized = res_high_lat.json()["latency_score"]
    print(f"   [OK] Latency 20ms Score: {lat_normal} | Latency 600ms Score: {lat_penalized} (clamped to 0)")
    assert lat_normal > 90.0 and lat_penalized == 0.0, "Latency scoring failed penalty test"

    # Edge Case B: Data Quality Degradation (missing values)
    res_clean = requests.post(f"{ML_URL}/arc/calculate", headers=ml_headers, json={
        "submission_id": "test-edge-3",
        "client_id": "client-test",
        "round_id": "round-test",
        "quality_metrics": {"missing_ratio": 0.0, "inf_ratio": 0.0, "preprocessing_success": 1.0}
    })
    res_dirty = requests.post(f"{ML_URL}/arc/calculate", headers=ml_headers, json={
        "submission_id": "test-edge-4",
        "client_id": "client-test",
        "round_id": "round-test",
        "quality_metrics": {"missing_ratio": 0.35, "inf_ratio": 0.05, "preprocessing_success": 1.0}
    })
    q_clean = res_clean.json()["quality_score"]
    q_dirty = res_dirty.json()["quality_score"]
    print(f"   [OK] Clean Data Quality: {q_clean} | 40% Missing/Inf Quality: {q_dirty}")
    assert q_clean == 100.0 and q_dirty == 60.0, "Quality scoring calculation mismatch"

    # 3. Dynamic Weight Configuration: Validation rejection
    print("\n[Step 3] Testing Dynamic Weight Configuration Validation...")
    res_invalid = requests.put(f"{BASE_URL}/arc/config", headers=headers, json={
        "lambdaRelevance": 0.50,
        "lambdaResource": 0.40,
        "lambdaCommunication": 0.30  # Sum = 1.20 != 1.0
    })
    print(f"   [OK] Invalid weights (sum=1.20) rejected with HTTP {res_invalid.status_code}: {res_invalid.json().get('message')}")
    assert res_invalid.status_code == 400, "Server failed to reject invalid lambda weights sum"

    # 4. Save Valid Dynamic Weights (emphasizing Relevance to 0.50)
    print("\n[Step 4] Updating Active ARC Weights (lambda_R=0.50, lambda_S=0.25, lambda_C=0.25)...")
    res_update = requests.put(f"{BASE_URL}/arc/config", headers=headers, json={
        "lambdaRelevance": 0.50,
        "lambdaResource": 0.25,
        "lambdaCommunication": 0.25,
        "recalculatePending": True
    })
    assert res_update.status_code == 200, f"Config update failed: {res_update.text}"
    cfg_data = res_update.json()
    new_version = cfg_data["config"]["version"]
    recalc = cfg_data.get("recalculation")
    print(f"   [OK] Config updated to version {new_version}.")
    if recalc:
        print(f"   [OK] Auto-recalculated {recalc['recalculatedCount']} submissions during update.")
        for r in recalc.get("results", []):
            print(f"        Client {r['clientId']}: {r['previousScore']} -> {r['newScore']} (Delta: {r['delta']})")

    # 5. Verify Configuration History
    print("\n[Step 5] Querying Formula Version History...")
    res_hist = requests.get(f"{BASE_URL}/arc/configs", headers=headers)
    assert res_hist.status_code == 200, f"History query failed: {res_hist.text}"
    configs = res_hist.json()["configs"]
    print(f"   [OK] Retrieved {len(configs)} configuration revision(s).")
    active_cfg = next((c for c in configs if c["isActive"]), None)
    assert active_cfg is not None, "No active config found in history"
    print(f"   [OK] Active revision: v{active_cfg['version']} (lambda_R: {active_cfg['lambdaRelevance']})")

    # 6. Verify Explicit Recalculation API
    print("\n[Step 6] Testing Explicit Recalculation Endpoint (/api/arc/recalculate)...")
    res_recalc = requests.post(f"{BASE_URL}/arc/recalculate", headers=headers, json={})
    assert res_recalc.status_code == 200, f"Recalculation failed: {res_recalc.text}"
    recalc_info = res_recalc.json()
    print(f"   [OK] Explicit recalculation succeeded: {recalc_info['message']}")
    for r in recalc_info.get("results", []):
        print(f"        - Client: {r['clientId']} | New Overall Score: {r['newScore']} | R={r['relevance']} S={r['resource']} C={r['communication']}")

    # 7. Verify Audit Log
    print("\n[Step 7] Checking Administrative Audit Logs for Weight Modifications...")
    res_logs = requests.get(f"{BASE_URL}/admin/audit-logs", headers=headers)
    assert res_logs.status_code == 200, f"Audit log query failed: {res_logs.text}"
    logs = res_logs.json()["logs"]
    arc_logs = [l for l in logs if l["action"] in ["UPDATE_ARC_CONFIG", "RECALCULATE_ARC"]]
    print(f"   [OK] Found {len(arc_logs)} ARC-related audit log entries:")
    for l in arc_logs[:3]:
        print(f"        • Action: {l['action']} | Resource: {l['resourceType']} | Timestamp: {l['timestamp']}")

    # 8. Reset to baseline v1.0 weights
    print("\n[Step 8] Resetting weights to baseline v1.0 (lambda_R=0.40, lambda_S=0.30, lambda_C=0.30)...")
    res_reset = requests.put(f"{BASE_URL}/arc/config", headers=headers, json={
        "lambdaRelevance": 0.40,
        "lambdaResource": 0.30,
        "lambdaCommunication": 0.30,
        "recalculatePending": True
    })
    assert res_reset.status_code == 200
    print(f"   [OK] Reset complete: active version is now v{res_reset.json()['config']['version']}.")

    print("\n" + "=" * 70)
    print("[SUCCESS] PHASE 5 ARC SCORING & CONFIGURATION FULLY VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
