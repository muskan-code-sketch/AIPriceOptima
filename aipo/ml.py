"""Model loading, prediction and honest held-out evaluation."""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data import DATA_PATH, ROOT, add_time_features

REQUIRED_INPUT_COLUMNS = ["OrderDate", "UnitPrice", "Discount", "Quantity"]


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Return (xgb_model, lgb_model, feature_columns) or raise a readable error."""
    try:
        xgb_json = ROOT / "xgboost_revenue_model.json"  # version-proof export, preferred
        if xgb_json.exists():
            import xgboost as xgb

            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(xgb_json))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(ROOT / "xgboost_revenue_model.pkl", "rb") as f:
                    xgb_model = pickle.load(f)
        with open(ROOT / "lightgbm_revenue_model.pkl", "rb") as f:
            lgb_model = pickle.load(f)
        with open(ROOT / "feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        raise RuntimeError(f"Could not load the trained models: {exc}") from exc
    return xgb_model, lgb_model, list(feature_columns)


def predict(models, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict revenue for rows that already have the engineered features."""
    xgb_model, lgb_model, cols = models
    x = frame[cols]
    out = pd.DataFrame({"XGBoost": xgb_model.predict(x), "LightGBM": lgb_model.predict(x)}, index=frame.index)
    out["Ensemble"] = out[["XGBoost", "LightGBM"]].mean(axis=1)
    return out


def predict_raw(models, raw: pd.DataFrame) -> pd.DataFrame:
    """Predict from raw order columns (OrderDate, UnitPrice, Discount, Quantity)."""
    return predict(models, add_time_features(raw))


def metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    return {
        "R²": r2_score(actual, pred),
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAPE": float(np.mean(np.abs((actual - pred) / actual)) * 100),
    }


@st.cache_data(show_spinner="Evaluating models on held-out orders...")
def heldout_report() -> dict:
    """Re-create the notebook's 80/20 split (random_state=42) and score both models."""
    models = load_models()
    xgb_model, lgb_model, cols = models
    df = add_time_features(pd.read_csv(DATA_PATH))
    x, y = df[cols], df["TotalAmount"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

    rows, preds = [], {}
    for split, xs, ys in (("Test (unseen)", x_te, y_te), ("Train", x_tr, y_tr)):
        p = predict(models, df.loc[xs.index])
        preds[split] = p
        for name in ("XGBoost", "LightGBM", "Ensemble"):
            rows.append({"Split": split, "Model": name, **metrics(ys, p[name])})

    test = df.loc[x_te.index, ["OrderDate", "Month", "UnitPrice", "Discount", "Quantity", "TotalAmount"]].copy()
    test = test.join(preds["Test (unseen)"])
    test["Error"] = test["Ensemble"] - test["TotalAmount"]

    imp = pd.DataFrame({
        "Feature": cols,
        "XGBoost": xgb_model.feature_importances_ / max(xgb_model.feature_importances_.sum(), 1e-12),
        "LightGBM": lgb_model.feature_importances_ / max(lgb_model.feature_importances_.sum(), 1e-12),
    })
    imp["Average"] = imp[["XGBoost", "LightGBM"]].mean(axis=1)
    return {"table": pd.DataFrame(rows), "test": test.reset_index(drop=True),
            "importance": imp.sort_values("Average", ascending=False), "n_test": len(x_te), "n_train": len(x_tr)}
