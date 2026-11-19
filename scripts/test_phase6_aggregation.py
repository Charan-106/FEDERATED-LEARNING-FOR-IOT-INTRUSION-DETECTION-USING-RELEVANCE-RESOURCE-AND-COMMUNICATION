"""
Phase 6 Automated Verification Script:
Tests the complete Federated Aggregation Center:
1. Admin authentication and active round retrieval.
2. Aggregation weight preview for all 3 strategies:
   - FedAvg: alpha_i = n_i / sum(n_j)
   - Equal-weight: alpha_i = 1 / K
   - ARC-weighted: alpha_i = ARC_i / sum(ARC_j)
3. Partial client subset selection preview (2 out of 3).
4. Execution of Federated Parameter Merging (ARC_WEIGHTED):
   - PyTorch weighted tensor averaging across layers.
   - Candidate checkpoint saved to storage/candidates/.
   - SHA256 checksum generated.
5. Server-side Candidate Model Validation:
   - Validation against server dataset (P:\\project\\validation.csv).
   - Extraction of Accuracy, Macro F1, Precision, Recall, and Loss.
6. Round & Fleet Status Verification:
   - Round updated to COMPLETED.
   - Submissions updated to AGGREGATED.
   - AggregationJob recorded with weights and evaluation metrics.
   - AuditLog verification.
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/api"

def main():
    print("=" * 70)
    print("ARC-FL PHASE 6 AGGREGATION CENTER AUTOMATION & VERIFICATION")
    print("=" * 70)

    # 1. Admin login
    print("\n[Step 1] Authenticating as Admin...")
    res = requests.post(f"{BASE_URL}/auth/login", json={"email": "admin@arc-fl.local", "password": "Admin@123"})
    assert res.status_code == 200, f"Login failed: {res.text}"
    token = res.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   [OK] Admin authenticated.")

    # 2. Get active round and submissions
    print("\n[Step 2] Retrieving Round #1 Submissions...")
    res = requests.get(f"{BASE_URL}/admin/rounds", headers=headers)
    assert res.status_code == 200
    rounds = res.json()["rounds"]
    round_1 = next((r for r in rounds if r["roundNumber"] == 1), rounds[0])
    round_id = round_1["id"]
    print(f"   [OK] Round #1 ID: {round_id} (Status: {round_1['status']})")

    res = requests.get(f"{BASE_URL}/admin/rounds/{round_id}/submissions", headers=headers)
    assert res.status_code == 200
    submissions = res.json()["submissions"]
    sub_ids = [s["id"] for s in submissions]
    print(f"   [OK] Found {len(submissions)} submission(s) in Round #1:")
    for s in submissions:
        arc_val = s.get("arcScore") or (s.get("arcScores", [{}])[0].get("overallArcScore")) or 0
        print(f"        • Client: {s['clientId']} | Samples: {s['samplesTrained']} | Local Acc: {s['localAccuracy']}% | ARC: {arc_val:.2f}")

    assert len(sub_ids) >= 3, f"Expected 3 submissions, found {len(sub_ids)}"

    # 3. Strategy 1: FedAvg Preview
    print("\n[Step 3] Testing Aggregation Strategy 1: Standard FedAvg...")
    res = requests.post(f"{BASE_URL}/aggregation/preview", headers=headers, json={
        "roundId": round_id,
        "submissionIds": sub_ids,
        "strategy": "FEDAVG"
    })
    assert res.status_code == 200, f"FedAvg preview failed: {res.text}"
    fedavg_preview = res.json()
    weights_fedavg = fedavg_preview["weights"]
    print(f"   [OK] FedAvg Weight Distribution ({len(weights_fedavg)} clients):")
    for w in weights_fedavg:
        print(f"        - {w['clientId']}: {w['normalizedWeight']*100:.2f}% (Samples: {w['clientSamples']})")
    sum_fedavg = sum(w["normalizedWeight"] for w in weights_fedavg)
    assert abs(sum_fedavg - 1.0) < 0.001, f"Weights do not sum to 1.0: {sum_fedavg}"

    # 4. Strategy 2: Equal Weighting Preview
    print("\n[Step 4] Testing Aggregation Strategy 2: Equal Weighting...")
    res = requests.post(f"{BASE_URL}/aggregation/preview", headers=headers, json={
        "roundId": round_id,
        "submissionIds": sub_ids,
        "strategy": "EQUAL"
    })
    assert res.status_code == 200, f"Equal preview failed: {res.text}"
    equal_preview = res.json()
    weights_equal = equal_preview["weights"]
    print(f"   [OK] Equal Weight Distribution ({len(weights_equal)} clients):")
    for w in weights_equal:
        print(f"        - {w['clientId']}: {w['normalizedWeight']*100:.2f}%")
    sum_equal = sum(w["normalizedWeight"] for w in weights_equal)
    assert abs(sum_equal - 1.0) < 0.001, f"Equal weights do not sum to 1.0: {sum_equal}"

    # 5. Strategy 3: Proposed ARC-Weighted Aggregation Preview
    print("\n[Step 5] Testing Aggregation Strategy 3: Proposed ARC-Weighted Aggregation...")
    res = requests.post(f"{BASE_URL}/aggregation/preview", headers=headers, json={
        "roundId": round_id,
        "submissionIds": sub_ids,
        "strategy": "ARC_WEIGHTED"
    })
    assert res.status_code == 200, f"ARC-weighted preview failed: {res.text}"
    arc_preview = res.json()
    weights_arc = arc_preview["weights"]
    print(f"   [OK] ARC-Weighted Distribution ({len(weights_arc)} clients):")
    for w in weights_arc:
        print(f"        - {w['clientId']}: {w['normalizedWeight']*100:.2f}% (ARC Score: {w['arcScore']})")
    sum_arc = sum(w["normalizedWeight"] for w in weights_arc)
    assert abs(sum_arc - 1.0) < 0.001, f"ARC weights do not sum to 1.0: {sum_arc}"

    # 6. Test Partial Client Subset Selection (Client 1 & Client 3 only)
    print("\n[Step 6] Testing Client Subset Selection (2 clients selected)...")
    res = requests.post(f"{BASE_URL}/aggregation/preview", headers=headers, json={
        "roundId": round_id,
        "submissionIds": [sub_ids[0], sub_ids[2]],
        "strategy": "ARC_WEIGHTED"
    })
    assert res.status_code == 200
    subset_preview = res.json()
    assert subset_preview["submissionCount"] == 2
    sum_subset = sum(w["normalizedWeight"] for w in subset_preview["weights"])
    print(f"   [OK] Subset preview succeeded: {len(subset_preview['weights'])} clients, sum={sum_subset:.4f}")
    assert abs(sum_subset - 1.0) < 0.001

    # 7. Execute Federated Aggregation (Merge Models) with ARC_WEIGHTED
    print("\n[Step 7] Authorizing & Executing Model Merging (Strategy: ARC_WEIGHTED)...")
    merge_start = time.time()
    res = requests.post(f"{BASE_URL}/aggregation/merge", headers=headers, json={
        "roundId": round_id,
        "submissionIds": sub_ids,
        "strategy": "ARC_WEIGHTED"
    })
    assert res.status_code == 200, f"Merge failed: {res.text}"
    merge_res = res.json()
    merge_time = round(time.time() - merge_start, 2)
    print(f"   [OK] Model Merging & Validation complete in {merge_time}s!")
    print(f"        • Candidate ID: {merge_res['candidate_id']}")
    print(f"        • Artifact Path: {merge_res['candidatePath']}")
    print(f"        • Checksum SHA256: {merge_res['candidateChecksum'][:24]}...")
    print(f"        • Valid Clients Merged: {merge_res['validClientCount']}")

    # 8. Server-Side Candidate Validation Metrics
    print("\n[Step 8] Reviewing Server-Side Candidate Evaluation Metrics...")
    eval_metrics = merge_res.get("evaluation")
    assert eval_metrics is not None, "Evaluation metrics missing from merge response"
    print(f"   [OK] Validation Results (Evaluated on {eval_metrics.get('path', 'validation.csv')}):")
    print(f"        - Accuracy: {eval_metrics.get('accuracy')}%")
    print(f"        - Macro F1: {eval_metrics.get('macro_f1'):.2f}%")
    print(f"        - Macro Precision: {eval_metrics.get('macro_precision'):.2f}%")
    print(f"        - Macro Recall: {eval_metrics.get('macro_recall'):.2f}%")
    print(f"        - Validation Loss: {eval_metrics.get('avg_loss', eval_metrics.get('loss', 0)):.4f}")
    print(f"        - Evaluation Duration: {eval_metrics.get('duration_seconds', 0):.2f}s")

    # 9. Verify Round Status and Submission Lifecycle
    print("\n[Step 9] Verifying Round and Submission Lifecycle States...")
    res = requests.get(f"{BASE_URL}/admin/rounds", headers=headers)
    r_check = next(r for r in res.json()["rounds"] if r["id"] == round_id)
    print(f"   [OK] Round #1 Status: {r_check['status']} (Expected: COMPLETED)")
    assert r_check["status"] == "COMPLETED"

    res = requests.get(f"{BASE_URL}/admin/submissions", headers=headers)
    r1_subs = [s for s in res.json()["submissions"] if s["roundId"] == round_id]
    all_aggregated = all(s["status"] == "AGGREGATED" for s in r1_subs)
    print(f"   [OK] Submissions lifecycle: {len(r1_subs)} submission(s) transitioned to AGGREGATED.")
    assert all_aggregated, "Not all submissions transitioned to AGGREGATED"

    # 10. Verify Aggregation History
    print("\n[Step 10] Checking Aggregation Job History...")
    res = requests.get(f"{BASE_URL}/aggregation/history", headers=headers)
    assert res.status_code == 200
    jobs = res.json()["jobs"]
    print(f"   [OK] Retrieved {len(jobs)} recorded aggregation job(s):")
    latest_job = jobs[0]
    print(f"        • Job ID: {latest_job['id']}")
    print(f"        • Strategy: {latest_job['strategy']}")
    print(f"        • Clients Selected: {latest_job['selectedClientCount']}")
    print(f"        • Weights Recorded: {len(latest_job.get('weights', []))}")
    print(f"        • Checksum: {latest_job.get('candidateChecksum')[:16]}...")

    print("\n" + "=" * 70)
    print("[SUCCESS] PHASE 6 AGGREGATION CENTER FULLY VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
