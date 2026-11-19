import pickle
import traceback

print("--- Testing label_encoder.pkl ---")
try:
    with open("P:/FLmodel/label_encoder.pkl", "rb") as f:
        data = f.read()
        print(f"File size: {len(data)} bytes")
        print("First 64 bytes:", data[:64])
        f.seek(0)
        le = pickle.load(f)
        print("Loaded successfully!", type(le))
        print("Classes count:", len(le.classes_))
        print("Classes:", le.classes_)
except Exception as e:
    traceback.print_exc()

print("\n--- Testing scaler.pkl ---")
try:
    with open("P:/FLmodel/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        print("Loaded scaler successfully!", type(scaler))
        print("Features count:", scaler.n_features_in_)
except Exception as e:
    traceback.print_exc()
