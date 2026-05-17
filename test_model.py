import joblib
import numpy as np

def test_model_loads():
    model = joblib.load("ml_model/model.pkl")
    assert model is not None, "Model failed to load"
    print("test_model_loads PASSED")

def test_scaler_loads():
    scaler = joblib.load("ml_model/scaler.pkl")
    assert scaler is not None, "Scaler failed to load"
    print("test_scaler_loads PASSED")

def test_threshold_loads():
    threshold = joblib.load("ml_model/threshold.pkl")
    assert 0.0 < threshold < 1.0, f"Threshold out of range: {threshold}"
    print(f"test_threshold_loads PASSED — threshold = {round(threshold, 4)}")

def test_model_predicts():
    model = joblib.load("ml_model/model.pkl")
    scaler = joblib.load("ml_model/scaler.pkl")
    feature_names = joblib.load("ml_model/feature_names.pkl")

    dummy = np.zeros((1, len(feature_names)))
    dummy_scaled = scaler.transform(dummy)
    proba = model.predict_proba(dummy_scaled)[0][1]

    assert 0.0 <= proba <= 1.0, f"Probability out of range: {proba}"
    print(f"test_model_predicts PASSED — dummy probability: {round(proba, 4)}")

def test_feature_names():
    feature_names = joblib.load("ml_model/feature_names.pkl")
    assert len(feature_names) > 0, "Feature names list is empty"
    print(f"test_feature_names PASSED — {len(feature_names)} features found")

if __name__ == "__main__":
    test_model_loads()
    test_scaler_loads()
    test_threshold_loads()
    test_model_predicts()
    test_feature_names()