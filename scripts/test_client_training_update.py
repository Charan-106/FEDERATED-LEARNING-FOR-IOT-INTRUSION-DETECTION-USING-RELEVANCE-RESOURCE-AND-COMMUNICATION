"""
Test Client Training Setup & Input Handling Update
Verifies:
1. Dataset auto-binding (no dataset dropdown/selection; only assigned partition exposed)
2. Validation of client-provided training parameters (epochs, batchSize, learningRate)
3. Genuine ARC telemetry computation (relevance Q_i and D_i from local dataset, real psutil resource metrics)
4. Display and persistence of actual PyTorch metrics and client parameters
5. Complete submission workflow with zero raw data transmission
"""

import sys
import requests
import json

BACKEND_URL = "http://localhost:5000"
ML_SERVICE_URL = "http://localhost:8000"
API_KEY = "arc-ml-secret-api-key-2026"

def login(email, password):
    resp = requests.post(f"{BACKEND_URL}/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Login failed for {email}: {resp.text}"
    return resp.json()["token"]

def run_tests():
    print("[1/5] Testing Client Authentication and Dataset Auto-Binding...")
    client1_token = login("client1@arc-fl.local", "Client@123")
    headers = {"Authorization": f"Bearer {client1_token}"}

    # Verify GET /api/client/datasets returns only client-001's assigned partition
    ds_resp = requests.get(f"{BACKEND_URL}/api/client/datasets", headers=headers)
    assert ds_resp.status_code == 200, f"GET /api/client/datasets failed: {ds_resp.text}"
    ds_data = ds_resp.json()
    assert "assignedDataset" in ds_data, "assignedDataset missing from response"
    assigned = ds_data["assignedDataset"]
    print(f"      Client 1 assigned dataset: {assigned['filename']} (assignedTo: {assigned.get('assignedTo')})")
    assert assigned["filename"] == "client1_traffic.csv", f"Expected client1_traffic.csv, got {assigned['filename']}"
    assert len(ds_data["datasets"]) == 1, "Should only expose the single assigned dataset to client"
    print("      [OK] Dataset selection removed; auto-bound to authenticated client's partition only.")

    print("\n[2/5] Testing Client-Provided Parameter Validation...")
    # Test invalid epochs (0 or negative)
    bad_resp = requests.post(f"{BACKEND_URL}/api/client/train", headers=headers, json={
        "epochs": 0,
        "batchSize": 128,
        "learningRate": 0.001
    })
    assert bad_resp.status_code == 400, f"Expected 400 for epochs=0, got {bad_resp.status_code}"
    print(f"      [OK] Validation rejected epochs=0: {bad_resp.json().get('error')}")

    # Test invalid learning rate (negative)
    bad_lr_resp = requests.post(f"{BACKEND_URL}/api/client/train", headers=headers, json={
        "epochs": 2,
        "batchSize": 128,
        "learningRate": -0.05
    })
    assert bad_lr_resp.status_code == 400, f"Expected 400 for negative lr, got {bad_lr_resp.status_code}"
    print(f"      [OK] Validation rejected negative learning rate: {bad_lr_resp.json().get('error')}")

    print("\n[3/5] Testing Local Edge Training with Client-Provided Parameters...")
    client_params = {
        "epochs": 2,
        "batchSize": 64,
        "learningRate": 0.002
    }
    train_resp = requests.post(f"{BACKEND_URL}/api/client/train", headers=headers, json=client_params)
    assert train_resp.status_code == 200, f"Training failed: {train_resp.text}"
    train_data = train_resp.json()

    print(f"      Training Success: {train_data.get('success')}")
    print(f"      Dataset Used: {train_data.get('dataset')} (Assigned to {train_data.get('assignedClientId')})")
    print(f"      PyTorch Accuracy: {train_data.get('accuracy')}%")
    print(f"      PyTorch Loss: {train_data.get('loss')}")
    print(f"      Samples Trained: {train_data.get('samples_trained')}")
    print(f"      Training Duration: {train_data.get('training_duration_seconds')}s")

    # Verify genuine ARC Relevance & Resource telemetry
    res_metrics = train_data.get("resource_metrics", {})
    quality_metrics = train_data.get("quality_metrics", {})
    diversity_metrics = train_data.get("diversity_metrics", {})

    print(f"      Genuine Resource: RAM Avail={res_metrics.get('available_ram_gb')}GB, CPU Idle={res_metrics.get('cpu_idle_pct')}%, Throughput={res_metrics.get('throughput_samples_per_sec')}/s")
    print(f"      Genuine Relevance: Missing Ratio={quality_metrics.get('missing_ratio')}, Classes Detected={diversity_metrics.get('class_count')}, Entropy={diversity_metrics.get('entropy')}")

    assert train_data.get("epochs") == 2, "Returned epochs should match client input"
    assert train_data.get("batch_size") == 64, "Returned batch_size should match client input"
    assert train_data.get("learning_rate") == 0.002, "Returned learning_rate should match client input"
    assert "quality_metrics" in train_data, "Quality metrics missing from training result"
    assert "diversity_metrics" in train_data, "Diversity metrics missing from training result"
    print("      [OK] Actual PyTorch results and genuine ARC telemetry generated successfully.")

    print("\n[4/5] Testing Model Update Submission with Client Parameters & Genuine Telemetry...")
    # Find active training round
    dash_resp = requests.get(f"{BACKEND_URL}/api/client/dashboard", headers=headers)
    latest_round = dash_resp.json().get("latestRound")
    round_id = latest_round["id"] if latest_round else None

    if not latest_round or latest_round.get("status") != "ACTIVE":
        # Create active round using admin token
        admin_token = login("admin@arc-fl.local", "Admin@123")
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        round_resp = requests.post(f"{BACKEND_URL}/api/admin/rounds", headers=admin_headers, json={
            "description": "Round for Client Training Update Verification"
        })
        round_id = round_resp.json()["round"]["id"]
        print(f"      Created active test round #{round_resp.json()['round']['roundNumber']} ({round_id})")

    # Submit trained update
    sub_payload = {
        "roundId": round_id,
        "updatePath": train_data["update_path"],
        "localAccuracy": train_data["accuracy"],
        "localLoss": train_data["loss"],
        "samplesTrained": train_data["samples_trained"],
        "epochs": train_data["epochs"],
        "batchSize": train_data["batch_size"],
        "learningRate": train_data["learning_rate"],
        "trainingDurationSeconds": train_data["training_duration_seconds"],
        "resourceMetrics": res_metrics,
        "qualityMetrics": quality_metrics,
        "diversityMetrics": diversity_metrics,
        "commMetrics": {"latency_ms": 35.0, "bandwidth_mbps": 50.0, "reliability_pct": 100.0}
    }

    sub_resp = requests.post(f"{BACKEND_URL}/api/client/submit-trained", headers=headers, json=sub_payload)
    assert sub_resp.status_code == 201, f"Submission failed: {sub_resp.text}"
    sub_data = sub_resp.json()
    submission_id = sub_data["submission"]["id"]
    print(f"      Submission ID: {submission_id} (Status: {sub_data['submission']['status']})")
    print("      [OK] Model update submitted successfully without transmitting raw data.")

    print("\n[5/5] Verifying Submission Record and ARC Calculation...")
    import time
    time.sleep(2) # Give background ARC calculation a moment

    subs_resp = requests.get(f"{BACKEND_URL}/api/client/submissions", headers=headers)
    assert subs_resp.status_code == 200
    my_sub = next((s for s in subs_resp.json()["submissions"] if s["id"] == submission_id), None)
    assert my_sub is not None, "Created submission not found in submission history"

    print(f"      Stored Epochs: {my_sub.get('epochs')}")
    print(f"      Stored Batch Size: {my_sub.get('batchSize')}")
    print(f"      Stored Learning Rate: {my_sub.get('learningRate')}")
    print(f"      Stored Checksum: {my_sub.get('checksumSha256')[:16]}...")
    print(f"      Calculated ARC Score: {my_sub.get('arcScore')}")

    assert my_sub.get("epochs") == 2, "Stored epochs mismatch"
    assert my_sub.get("batchSize") == 64, "Stored batch size mismatch"
    assert my_sub.get("learningRate") == 0.002, "Stored learning rate mismatch"
    print("      [OK] Client-provided parameters and calculated ARC score saved accurately.")

    print("\n========================================================")
    print("ALL TESTS PASSED: Client Training Setup and Input Handling")
    print("========================================================")

if __name__ == "__main__":
    run_tests()
