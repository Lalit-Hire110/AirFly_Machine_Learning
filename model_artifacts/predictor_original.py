
import json
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path

ART_DIR = Path("model_artifacts")

# load artifacts
clf = load(ART_DIR / "clf_pipe.joblib")
reg_norm = load(ART_DIR / "normal_reg_pipe.joblib")
reg_heavy = load(ART_DIR / "heavy_reg_pipe.joblib")
with open(ART_DIR / "features.json", "r") as f:
    features = json.load(f)

HEAVY_THRESH = 30.0

def predict_single(row_dict):
    """
    row_dict: mapping of feature_name -> value (must include categorical keys and numeric keys used by the model)
    Returns: dict with predicted_delay (minutes) and heavy_flag (0/1)
    """
    # build DataFrame
    df = pd.DataFrame([row_dict])
    X = df[features['numeric_features'] + features['categorical_features']]

    heavy_prob = clf.predict(X)[0]
    # regressors expect the whole feature set; they output log1p predictions
    pred_norm_log = reg_norm.predict(X)[0]
    pred_heavy_log = reg_heavy.predict(X)[0]
    pred_norm = float(np.expm1(pred_norm_log))
    pred_heavy = float(np.expm1(pred_heavy_log))
    pred = pred_heavy if heavy_prob == 1 else pred_norm
    pred = max(0.0, pred)
    return {
        "predicted_delay_min": pred,
        "heavy_flag": int(heavy_prob),
        "pred_norm": pred_norm,
        "pred_heavy": pred_heavy
    }
