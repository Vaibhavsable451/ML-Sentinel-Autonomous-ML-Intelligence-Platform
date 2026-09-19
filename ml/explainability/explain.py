"""Explainability: global feature importance, per-instance SHAP, and counterfactuals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def global_shap_importance(model, X_background: np.ndarray, feature_names: list[str], max_samples: int = 300) -> pd.Series:
    """Model-agnostic global importance using SHAP (falls back to permutation importance
    only if a fast explainer isn't applicable)."""
    sample = X_background[:max_samples]
    try:
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
        vals = np.abs(shap_values.values)
        if vals.ndim == 3:  # multi-class output -> take positive class
            vals = vals[:, :, -1]
        importance = vals.mean(axis=0)
    except Exception:  # noqa: BLE001
        from sklearn.inspection import permutation_importance
        y_dummy = model.predict(sample)
        result = permutation_importance(model, sample, y_dummy, n_repeats=5, random_state=42)
        importance = result.importances_mean
    return pd.Series(importance, index=feature_names).sort_values(ascending=False)


def explain_instance(model, x_row: np.ndarray, X_background: np.ndarray, feature_names: list[str], top_k: int = 5) -> list[dict]:
    """Returns the top-k contributing features for a single prediction."""
    try:
        explainer = shap.Explainer(model, X_background[:200])
        sv = explainer(x_row.reshape(1, -1))
        vals = sv.values[0]
        if vals.ndim == 2:
            vals = vals[:, -1]
    except Exception:  # noqa: BLE001
        base_prob = model.predict_proba(x_row.reshape(1, -1))[0, 1]
        vals = np.zeros(len(feature_names))
        means = X_background.mean(axis=0)
        for i in range(len(feature_names)):
            perturbed = x_row.copy()
            perturbed[i] = means[i]
            new_prob = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
            vals[i] = base_prob - new_prob

    order = np.argsort(-np.abs(vals))[:top_k]
    return [{"feature": feature_names[i], "contribution": round(float(vals[i]), 4), "value": round(float(x_row[i]), 3)} for i in order]


def clean_feature_label(name: str) -> str:
    """Helper to convert feature name to human readable string."""
    name = name.replace("num__", "").replace("cat__", "").replace("num_", "").replace("cat_", "")
    parts = name.split("_")
    if parts[0] == "contract" and len(parts) > 2:
        val = " ".join(parts[2:]).title()
        return f"Contract: {val}"
    if parts[0] == "payment" and len(parts) > 2:
        val = " ".join(parts[2:]).title()
        return f"Payment: {val}"
    if parts[0] == "plan" and len(parts) > 1:
        val = " ".join(parts[1:]).title()
        return f"Plan: {val}"
    if parts[0] == "region" and len(parts) > 1:
        val = " ".join(parts[1:]).title()
        return f"Region: {val}"
    res = " ".join(parts).title()
    res = res.replace("90D", "(90d)").replace("Months", "(Months)")
    return res


def find_counterfactual(
    model, x_row: np.ndarray, feature_names: list[str],
    target_class: int = 0, max_steps: int = 80, step_size: float = 0.25,
) -> dict:
    """Gradient-guided counterfactual generator.

    Iteratively nudges features along the gradient direction that reduces churn
    risk, producing actionable, human-readable prescriptions.
    """
    x_curr = x_row.copy()
    prob_orig = float(model.predict_proba(x_row.reshape(1, -1))[0, 1])

    if prob_orig < 0.5 and target_class == 0:
        return {
            "status": "already_target_class",
            "message": "Customer is already predicted as low risk (RETAIN).",
            "current_churn_probability": f"{prob_orig*100:.1f}%",
        }

    epsilon = 1e-3
    num_features = len(feature_names)

    for step in range(1, max_steps + 1):
        prob_curr = float(model.predict_proba(x_curr.reshape(1, -1))[0, 1])
        if prob_curr < 0.5:
            diffs = x_curr - x_row
            top_changed_indices = np.argsort(-np.abs(diffs))[:5]

            actionable_prescriptions = []
            for idx in top_changed_indices:
                if abs(diffs[idx]) > 1e-3:
                    fname = feature_names[idx]
                    clean_name = clean_feature_label(fname)
                    direction = "Increase" if diffs[idx] > 0 else "Decrease"
                    actionable_prescriptions.append(f"{direction} {clean_name}")

            return {
                "status": "prescription_found",
                "target_outcome": "RETAIN (Low Churn Risk)",
                "original_churn_probability": f"{prob_orig*100:.1f}%",
                "new_churn_probability": f"{prob_curr*100:.1f}%",
                "steps_taken": step,
                "prescriptions": actionable_prescriptions,
            }

        # Compute numerical gradient of churn probability
        grads = np.zeros(num_features)
        for i in range(num_features):
            x_p = x_curr.copy()
            x_p[i] += epsilon
            p_plus = float(model.predict_proba(x_p.reshape(1, -1))[0, 1])
            grads[i] = (p_plus - prob_curr) / epsilon

        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-6:
            grads = grads / grad_norm

        x_curr -= step_size * grads

    return {
        "status": "partial_recommendation",
        "original_churn_probability": f"{prob_orig*100:.1f}%",
        "reduced_churn_probability": f"{prob_curr*100:.1f}%",
        "message": "Significantly reduced risk, but customer profile requires comprehensive retention intervention.",
    }
