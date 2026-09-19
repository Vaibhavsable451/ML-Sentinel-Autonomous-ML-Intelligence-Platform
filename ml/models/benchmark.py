"""Algorithm benchmark engine: trains multiple classifiers and ranks them.

This is the "compare 9+ algorithms automatically" piece of the platform.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def _model_zoo(random_state: int = 42) -> dict:
    zoo = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1),
        "svm_rbf": SVC(probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=15),
    }
    if HAS_XGB:
        zoo["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
    if HAS_LGBM:
        zoo["lightgbm"] = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=random_state, verbosity=-1)
    if HAS_CATBOOST:
        zoo["catboost"] = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_state=random_state, verbose=False)
    return zoo


@dataclass
class BenchmarkResult:
    leaderboard: pd.DataFrame
    fitted_models: dict
    champion_name: str
    champion_model: object


def run_benchmark(
    X_train: np.ndarray, y_train: np.ndarray,
    cv_folds: int = 5, random_state: int = 42,
) -> BenchmarkResult:
    zoo = _model_zoo(random_state)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
    }

    rows = []
    fitted = {}
    for name, model in zoo.items():
        t0 = time.time()
        cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1, return_estimator=False)
        elapsed = time.time() - t0
        model.fit(X_train, y_train)  # final fit on full training set
        fitted[name] = model
        rows.append({
            "model": name,
            "cv_f1": round(cv_res["test_f1"].mean(), 4),
            "cv_roc_auc": round(cv_res["test_roc_auc"].mean(), 4),
            "cv_accuracy": round(cv_res["test_accuracy"].mean(), 4),
            "cv_precision": round(cv_res["test_precision"].mean(), 4),
            "cv_recall": round(cv_res["test_recall"].mean(), 4),
            "cv_std_f1": round(cv_res["test_f1"].std(), 4),
            "train_time_sec": round(elapsed, 2),
        })

    leaderboard = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    # ensemble: soft-voting stack of the top-3 base learners
    top3 = leaderboard["model"].head(3).tolist()
    voting = VotingClassifier(estimators=[(n, fitted[n]) for n in top3], voting="soft")
    voting.fit(X_train, y_train)
    fitted["voting_ensemble_top3"] = voting
    cv_res = cross_validate(voting, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    leaderboard = pd.concat([leaderboard, pd.DataFrame([{
        "model": "voting_ensemble_top3",
        "cv_f1": round(cv_res["test_f1"].mean(), 4),
        "cv_roc_auc": round(cv_res["test_roc_auc"].mean(), 4),
        "cv_accuracy": round(cv_res["test_accuracy"].mean(), 4),
        "cv_precision": round(cv_res["test_precision"].mean(), 4),
        "cv_recall": round(cv_res["test_recall"].mean(), 4),
        "cv_std_f1": round(cv_res["test_f1"].std(), 4),
        "train_time_sec": None,
    }])], ignore_index=True).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    champion_name = leaderboard.iloc[0]["model"]
    return BenchmarkResult(
        leaderboard=leaderboard,
        fitted_models=fitted,
        champion_name=champion_name,
        champion_model=fitted[champion_name],
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ml.preprocessing.pipeline import run_pipeline

    df = pd.read_csv("data/train_raw.csv")
    result = run_pipeline(df, target="churned")
    bench = run_benchmark(result.X_train, result.y_train, cv_folds=4)
    print(bench.leaderboard.to_string(index=False))
    print(f"\nChampion: {bench.champion_name}")

    from sklearn.metrics import classification_report
    y_pred = bench.champion_model.predict(result.X_test)
    print("\nHeld-out TEST set report:")
    print(classification_report(result.y_test, y_pred))
