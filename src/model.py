"""Training, evaluation, persistence and prediction for the stroke-risk model."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .config import (FEATURES, METRICS_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, RISK_MEDIUM_FRACTION)
from .preprocessing import build_preprocessor, split_xy


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _fit(name: str, clf, X_train, y_train) -> Pipeline:
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    if name == "Gradient Boosting":  # has no class_weight argument -> use sample weights
        pipe.fit(X_train, y_train, clf__sample_weight=compute_sample_weight("balanced", y_train))
    else:
        pipe.fit(X_train, y_train)
    return pipe


def best_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on a validation split. beta=2 favours recall (screening use:
    missing a real stroke case is worse than a false alarm)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    fb = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-9, None)
    return float(thr[int(np.argmax(fb))]) if len(thr) else 0.5


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float((pred == np.asarray(y_true)).mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_compare(df: pd.DataFrame, test_size: float = 0.2, save: bool = True) -> dict:
    """Train all candidates, pick the best by ROC-AUC on a validation split, report on a held-out test set."""
    X, y = split_xy(df)
    if y.nunique() < 2:
        raise ValueError("Target needs both classes (0 and 1) to train.")
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, clf in candidate_models().items():
        pipe = _fit(name, clf, X_tr, y_tr)
        thr = best_threshold(y_val, pipe.predict_proba(X_val)[:, 1])
        val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        results[name] = {"val_roc_auc": float(val_auc), "threshold": thr}
        fitted[name] = pipe
    best_name = max(results, key=lambda k: results[k]["val_roc_auc"])

    # refit the winner on train+validation, evaluate once on the untouched test set
    final = _fit(best_name, candidate_models()[best_name], X_tmp, y_tmp)
    test_proba = final.predict_proba(X_test)[:, 1]
    thr = results[best_name]["threshold"]
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    prec, rec, _ = precision_recall_curve(y_test, test_proba)

    # comparison table on the same test set (models trained on train split only)
    comparison = {}
    for name, pipe in fitted.items():
        p = pipe.predict_proba(X_test)[:, 1]
        comparison[name] = evaluate(y_test, p, results[name]["threshold"])

    report = {
        "best_model": best_name, "threshold": thr,
        "test_metrics": evaluate(y_test, test_proba, thr),
        "comparison": comparison, "validation": results,
        "n_train": int(len(X_tmp)), "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "curves": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": prec.tolist(), "recall": rec.tolist()},
        "importances": feature_importances(final),
    }
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": final, "threshold": thr, "best_model": best_name}, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


def feature_importances(pipe: Pipeline) -> dict:
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
    else:
        return {}
    top = sorted(zip(names, vals), key=lambda t: t[1], reverse=True)[:12]
    return {k: float(v) for k, v in top}


def model_exists() -> bool:
    return MODEL_PATH.exists()


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Open the 'Train Model' page first.")
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict | None:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


def risk_band(p: float, threshold: float) -> str:
    """Bands are relative to the decision threshold because class weighting inflates raw probabilities.
    High: at/above threshold. Medium: at/above half the threshold. Low: below."""
    if p >= threshold:
        return "High"
    return "Medium" if p >= threshold * RISK_MEDIUM_FRACTION else "Low"


def predict_one(bundle: dict, record: dict) -> dict:
    row = pd.DataFrame([record])[FEATURES]
    p = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {"probability": p, "band": risk_band(p, bundle["threshold"]), "flagged": p >= bundle["threshold"],
            "threshold": bundle["threshold"]}


def predict_many(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    p = bundle["pipeline"].predict_proba(df[FEATURES])[:, 1]
    out = df.copy()
    out["risk_probability"] = p.round(4)
    out["risk_band"] = [risk_band(x, bundle["threshold"]) for x in p]
    out["flagged"] = (p >= bundle["threshold"]).astype(int)
    return out
