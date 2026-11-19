import joblib
import traceback

print("--- Testing joblib.load on label_encoder.pkl ---")
try:
    le = joblib.load("P:/FLmodel/label_encoder.pkl")
    print("SUCCESS!", type(le))
    print("Classes count:", len(le.classes_))
    print("Classes:", le.classes_)
except Exception:
    traceback.print_exc()

print("\n--- Testing joblib.load on scaler.pkl ---")
try:
    scaler = joblib.load("P:/FLmodel/scaler.pkl")
    print("SUCCESS!", type(scaler))
    print("Features count:", scaler.n_features_in_)
    print("Mean shape:", scaler.mean_.shape)
    print("Scale shape:", scaler.scale_.shape)
except Exception:
    traceback.print_exc()
