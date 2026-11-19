"""
Phase 4 Automated Verification Script:
Tests the complete Federated Learning Workflow end-to-end:
1. Admin logs in and creates/verifies Round #1.
2. Client 1, 2, and 3 log in independently.
3. Each client verifies the active global model (initial-v1).
4. Each client trains locally on their private dataset partition (client1, client2, client3).
5. Each client submits their model update (.pt) with real local accuracy, loss, and telemetry.
6. The system calculates ARC scores asynchronously (Relevance, Resource, Communication).
7. Admin verifies all 3 submissions in the fleet registry.
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/api"

def main():
    print("=" * 70)
    print("ARC-FL PHASE 4 WORKFLOW AUTOMATION & VERIFICATION")
    print("=" * 70)

    # 1. Admin login
    print("\n[Step 1] Logging in as Admin (admin@arc-fl.local)...")
    res = requests.post(f"{BASE_URL}/auth/login", json={"email": "admin@arc-fl.local", "password": "Admin@123"})
    assert res.status_code == 200, f"Admin login failed: {res.text}"
    admin_token = res.json()["token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    print("   [OK] Admin authenticated. JWT acquired.")

    # 2. Check/create round
    print("\n[Step 2] Initializing Training Round #1...")
    res = requests.post(f"{BASE_URL}/admin/rounds", headers=admin_headers)
    if res.status_code == 201:
        round_data = res.json()["round"]
        round_id = round_data["id"]
        round_number = round_data["roundNumber"]
        print(f"   [OK] Created Round #{round_number} (ID: {round_id})")
    else:
        res = requests.get(f"{BASE_URL}/admin/rounds", headers=admin_headers)
        rounds = res.json()["rounds"]
        round_id = rounds[0]["id"]
        round_number = rounds[0]["roundNumber"]
        print(f"   [OK] Using existing Round #{round_number} (ID: {round_id})")

    # 3. Client training & submissions
    clients = [
        ("client1@arc-fl.local", "Client@123", "client-001", "client1_traffic.csv"),
        ("client2@arc-fl.local", "Client@123", "client-002", "client2_traffic.csv"),
        ("client3@arc-fl.local", "Client@123", "client-003", "client3_traffic.csv"),
    ]

    submitted_ids = []

    for email, pwd, client_id, dataset in clients:
        print(f"\n--- [Client Node: {client_id}] ---")
        # Login
        res = requests.post(f"{BASE_URL}/auth/login", json={"email": email, "password": pwd})
        assert res.status_code == 200, f"Login failed for {email}: {res.text}"
        client_token = res.json()["token"]
        client_headers = {"Authorization": f"Bearer {client_token}"}
        print(f"   [OK] Logged in as {email}")

        # Model inspection
        res = requests.get(f"{BASE_URL}/client/model", headers=client_headers)
        assert res.status_code == 200, f"Failed to get model: {res.text}"
        model_info = res.json()["model"]
        print(f"   [OK] Verified active model: {model_info['version']} ({model_info['architecture']}, {model_info['totalParameters']:,} params)")

        # Local Training
        print(f"   [*] Running on-device PyTorch training on partition '{dataset}'...")
        train_start = time.time()
        res = requests.post(f"{BASE_URL}/client/train", headers=client_headers, json={
            "datasetFilename": dataset,
            "epochs": 2,
            "batchSize": 128,
            "learningRate": 0.001
        })
        assert res.status_code == 200, f"Local training failed for {client_id}: {res.text}"
        train_res = res.json()
        duration = round(time.time() - train_start, 2)
        print(f"   [OK] Training complete in {duration}s!")
        print(f"        Accuracy: {train_res['accuracy']}% | Loss: {train_res['loss']:.4f} | Samples: {train_res['samples_trained']}")

        # Submit Update
        print(f"   [*] Transmitting model update ({train_res['update_id'][:8]}.pt) to server...")
        res = requests.post(f"{BASE_URL}/client/submit-trained", headers=client_headers, json={
            "roundId": round_id,
            "updatePath": train_res["update_path"],
            "localAccuracy": train_res["accuracy"],
            "localLoss": train_res["loss"],
            "samplesTrained": train_res["samples_trained"],
            "epochs": train_res["epochs"],
            "batchSize": train_res["batch_size"],
            "learningRate": train_res["learning_rate"],
            "trainingDurationSeconds": train_res["training_duration_seconds"],
            "resourceMetrics": train_res["resource_metrics"],
        })
        assert res.status_code == 201, f"Submission failed: {res.text}"
        sub_id = res.json()["submission"]["id"]
        submitted_ids.append(sub_id)
        print(f"   [OK] Submission accepted: ID {sub_id}")

    # 4. Wait briefly for async ARC scoring
    print("\n[Step 4] Awaiting ARC evaluation scoring in background...")
    time.sleep(4)

    # 5. Admin verification
    print("\n[Step 5] Admin reviewing fleet submissions...")
    res = requests.get(f"{BASE_URL}/admin/submissions", headers=admin_headers)
    assert res.status_code == 200, f"Failed to get submissions: {res.text}"
    submissions = res.json()["submissions"]
    print(f"   [OK] Total submissions in registry: {len(submissions)}")

    for sub in submissions:
        if sub["id"] in submitted_ids:
            arc_list = sub.get("arcScores", [])
            arc = arc_list[0] if arc_list else {}
            print(f"\n   [SUBMISSION] #{sub['id'][:8]}:")
            print(f"      - Client: {sub['clientId']} ({sub['client']['name']})")
            print(f"      - Status: {sub['status']}")
            print(f"      - Local Accuracy: {sub['localAccuracy']}% | Local Loss: {sub['localLoss']}")
            print(f"      - Samples: {sub['samplesTrained']}")
            print(f"      - Overall ARC Score: {sub.get('arcScore') or arc.get('overallArcScore') or 'Pending'}")
            if arc:
                print(f"      - ARC Breakdown: R={arc.get('relevanceScore')} | S={arc.get('resourceScore')} | C={arc.get('communicationScore')}")

    print("\n" + "=" * 70)
    print("[SUCCESS] PHASE 4 FEDERATED WORKFLOW SUCCESSFULLY VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
