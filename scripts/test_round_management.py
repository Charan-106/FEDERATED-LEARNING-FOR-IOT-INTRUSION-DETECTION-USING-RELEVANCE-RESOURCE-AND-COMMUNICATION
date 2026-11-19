"""
Test Suite: Complete Federated Round Management Lifecycle
Tests:
1. Admin authentication
2. Configurable round creation (name, clients, hyperparameters, ARC weights)
3. Input validation (rejection of invalid ARC weights and hyperparameters)
4. Full round details retrieval
5. Editing of pending (CREATED) rounds
6. Transition from CREATED to ACTIVE via Start Round
7. Cancellation of active/pending rounds
8. Safe deletion of disposable (CANCELLED/CREATED) rounds
9. Strict protection of COMPLETED rounds against deletion
"""

import requests
import json

BASE_URL = "http://localhost:5000/api"

def main():
    print("=" * 70)
    print("TESTING FEDERATED ROUND MANAGEMENT & LIFECYCLE CONTROLS")
    print("=" * 70)

    # 1. Admin login
    print("\n[Step 1] Authenticating Admin...")
    res = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "admin@arc-fl.local",
        "password": "Admin@123"
    })
    assert res.status_code == 200, f"Admin login failed: {res.text}"
    token = res.json()["token"]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    print("  [OK] Admin authenticated.")

    # 2. Test Input Validation Rejection
    print("\n[Step 2] Testing Configuration Validation & Rejection...")
    
    # 2a. Invalid ARC weights (sum != 1.0)
    invalid_arc_res = requests.post(f"{BASE_URL}/admin/rounds", headers=headers, json={
        "name": "Invalid ARC Round",
        "arcWeights": {"lambdaRelevance": 0.5, "lambdaResource": 0.5, "lambdaCommunication": 0.5}
    })
    assert invalid_arc_res.status_code == 400, "Failed to reject invalid ARC weights"
    print(f"  [OK] Correctly rejected invalid ARC weights (sum=1.5): {invalid_arc_res.json()['error']}")

    # 2b. Invalid epochs (0)
    invalid_epoch_res = requests.post(f"{BASE_URL}/admin/rounds", headers=headers, json={
        "name": "Invalid Epochs Round",
        "hyperparameters": {"epochs": 0, "batchSize": 32, "learningRate": 0.001}
    })
    assert invalid_epoch_res.status_code == 400, "Failed to reject invalid epochs"
    print(f"  [OK] Correctly rejected invalid local epochs: {invalid_epoch_res.json()['error']}")

    # 3. Create Configurable Round (Saved as CREATED / Pending)
    print("\n[Step 3] Creating Configured Pending Round...")
    create_payload = {
        "name": "Academic Review Round (One-Client)",
        "description": "Validation trial for CICIoT2023 preact_resmlp_v3 architecture",
        "clientIds": ["client-001"],
        "hyperparameters": {
            "epochs": 2,
            "batchSize": 64,
            "learningRate": 0.002,
            "timeoutSeconds": 300
        },
        "arcWeights": {
            "lambdaRelevance": 0.40,
            "lambdaResource": 0.30,
            "lambdaCommunication": 0.30
        },
        "aggregationStrategy": "ARC_WEIGHTED"
    }
    create_res = requests.post(f"{BASE_URL}/admin/rounds", headers=headers, json=create_payload)
    assert create_res.status_code == 201, f"Create round failed: {create_res.text}"
    round_data = create_res.json()["round"]
    round_id = round_data["id"]
    round_num = round_data["roundNumber"]
    
    assert round_data["status"] == "CREATED", f"Expected CREATED status, got {round_data['status']}"
    assert round_data["name"] == create_payload["name"]
    assert round_data["targetClients"] == ["client-001"]
    assert round_data["hyperparameters"]["epochs"] == 2
    assert round_data["arcWeights"]["lambdaRelevance"] == 0.40
    print(f"  [OK] Round #{round_num} created successfully with ID: {round_id}")
    print(f"       Status: {round_data['status']} (pending, does not auto-start)")
    print(f"       Target Clients: {round_data['targetClients']}")
    print(f"       Hyperparameters: {round_data['hyperparameters']}")
    print(f"       ARC Weights: {round_data['arcWeights']}")

    # 4. Fetch Round Details
    print("\n[Step 4] Fetching Complete Round Details...")
    details_res = requests.get(f"{BASE_URL}/admin/rounds/{round_id}", headers=headers)
    assert details_res.status_code == 200, f"Get round details failed: {details_res.text}"
    details = details_res.json()["round"]
    assert details["baseModel"]["version"] is not None
    assert details["baseModel"]["checksumSha256"] is not None
    print(f"  [OK] Retrieved round details. Base Model: {details['baseModel']['version']} (Hash: {details['baseModel']['checksumSha256'][:16]}...)")

    # 5. Edit Pending Round Configuration
    print("\n[Step 5] Editing Pending Round Configuration...")
    edit_payload = {
        "name": "Updated Academic Review Round",
        "description": "Refined description with updated batch size",
        "hyperparameters": {
            "epochs": 3,
            "batchSize": 128,
            "learningRate": 0.001,
            "timeoutSeconds": 600
        }
    }
    update_res = requests.put(f"{BASE_URL}/admin/rounds/{round_id}", headers=headers, json=edit_payload)
    assert update_res.status_code == 200, f"Update round failed: {update_res.text}"
    updated_round = update_res.json()["round"]
    assert updated_round["name"] == edit_payload["name"]
    assert updated_round["hyperparameters"]["epochs"] == 3
    assert updated_round["hyperparameters"]["batchSize"] == 128
    print(f"  [OK] Round successfully edited while in CREATED status.")
    print(f"       Updated Name: {updated_round['name']}")
    print(f"       Updated Hyperparams: {updated_round['hyperparameters']}")

    # 6. Start Round (Transition CREATED -> ACTIVE)
    print("\n[Step 6] Starting Round...")
    start_res = requests.post(f"{BASE_URL}/admin/rounds/{round_id}/start", headers=headers)
    assert start_res.status_code == 200, f"Start round failed: {start_res.text}"
    active_round = start_res.json()["round"]
    assert active_round["status"] == "ACTIVE"
    print(f"  [OK] Round #{round_num} started successfully. Status: {active_round['status']}")

    # 7. Verify Editing is BLOCKED Once Round is ACTIVE
    print("\n[Step 7] Verifying Immutability of ACTIVE Round...")
    blocked_edit_res = requests.put(f"{BASE_URL}/admin/rounds/{round_id}", headers=headers, json={"name": "Illegal Edit"})
    assert blocked_edit_res.status_code == 400, "Active round should not be editable"
    print(f"  [OK] Active round editing blocked: {blocked_edit_res.json()['error']}")

    # 8. Cancel Round
    print("\n[Step 8] Cancelling Active Round...")
    cancel_res = requests.post(f"{BASE_URL}/admin/rounds/{round_id}/cancel", headers=headers, json={"reason": "Testing cancellation"})
    assert cancel_res.status_code == 200, f"Cancel round failed: {cancel_res.text}"
    cancelled_round = cancel_res.json()["round"]
    assert cancelled_round["status"] == "CANCELLED"
    print(f"  [OK] Round #{round_num} cancelled. Status: {cancelled_round['status']}")

    # 9. Delete Disposable Cancelled Round
    print("\n[Step 9] Deleting Cancelled Round...")
    delete_res = requests.delete(f"{BASE_URL}/admin/rounds/{round_id}", headers=headers)
    assert delete_res.status_code == 200, f"Delete round failed: {delete_res.text}"
    print(f"  [OK] Round #{round_num} safely deleted.")

    # Verify it no longer exists
    not_found_res = requests.get(f"{BASE_URL}/admin/rounds/{round_id}", headers=headers)
    assert not_found_res.status_code == 404
    print("  [OK] Confirmed deletion from database.")

    # 10. Strict Protection of Completed Rounds
    print("\n[Step 10] Testing Strict Protection of COMPLETED Historical Rounds...")
    # Find Round #1 (COMPLETED)
    rounds_res = requests.get(f"{BASE_URL}/admin/rounds", headers=headers)
    completed_round = next((r for r in rounds_res.json()["rounds"] if r["status"] == "COMPLETED"), None)
    if completed_round:
        illegal_delete_res = requests.delete(f"{BASE_URL}/admin/rounds/{completed_round['id']}", headers=headers)
        assert illegal_delete_res.status_code == 400, "COMPLETED round must not be deletable"
        print(f"  [OK] Deletion of COMPLETED Round #{completed_round['roundNumber']} strictly blocked:")
        print(f"       '{illegal_delete_res.json()['error']}'")

    print("\n" + "=" * 70)
    print("ALL ROUND MANAGEMENT TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
