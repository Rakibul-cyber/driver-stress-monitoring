"""Train and evaluate a Random Forest stress classifier on WESAD HRV features.

Evaluation protocol (subject-independent, no leakage):
  1. Subjects 15, 16, 17 are held out completely (final test + replay data).
  2. Leave-one-subject-out CV on the remaining 12 subjects.
  3. Final model is fit on all 12 training subjects and scored on the held-out 3.

Usage:  python -m src.train
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURES,
    HOLDOUT_PARQUET,
    LABEL_NAMES,
    METADATA_PATH,
    MODEL_PATH,
    clean,
    load_raw,
)

SEED = 42
TEST_SUBJECTS = [15, 16, 17]
# Source windows are emitted every 0.25 s and overlap heavily; keep every 4th
# row (1 Hz) for training to reduce redundancy and training time.
TRAIN_STRIDE = 4
THRESHOLD = 0.5


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


def compute_metrics(y_true, y_prob) -> dict:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def stride(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.groupby("subject_id").cumcount() % TRAIN_STRIDE == 0]


def main() -> None:
    t0 = time.time()
    df = clean(load_raw())
    print(f"Clean dataset: {df.shape[0]} rows, {df.subject_id.nunique()} subjects")
    print(df.groupby("label").size().rename(LABEL_NAMES).to_string())

    train_df = df[~df.subject_id.isin(TEST_SUBJECTS)]
    test_df = df[df.subject_id.isin(TEST_SUBJECTS)]
    train_s = stride(train_df)
    X, y, groups = train_s[FEATURES], train_s["label"].to_numpy(), train_s["subject_id"]
    print(f"Train: {len(train_s)} rows (stride {TRAIN_STRIDE}), "
          f"{groups.nunique()} subjects | Test: {len(test_df)} rows, subjects {TEST_SUBJECTS}")

    # Leave-one-subject-out CV
    folds = []
    for tr, va in LeaveOneGroupOut().split(X, y, groups):
        model = build_pipeline().fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[va])[:, 1]
        m = compute_metrics(y[va], prob)
        m["subject_id"] = int(groups.iloc[va].iloc[0])
        folds.append(m)
        print(f"  LOSO subject {m['subject_id']:>2}: acc={m['accuracy']:.3f} f1={m['f1']:.3f}")
    keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]
    cv_summary = {
        k: {"mean": round(float(np.mean([f[k] for f in folds])), 4),
            "std": round(float(np.std([f[k] for f in folds])), 4)}
        for k in keys
    }

    # Final model on all training subjects, evaluated on held-out subjects
    model = build_pipeline().fit(X, y)
    test_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    test_metrics = compute_metrics(test_df["label"].to_numpy(), test_prob)
    per_subject_test = {
        int(s): compute_metrics(g["label"].to_numpy(), model.predict_proba(g[FEATURES])[:, 1])
        for s, g in test_df.groupby("subject_id")
    }

    importances = sorted(
        zip(FEATURES, model.named_steps["rf"].feature_importances_),
        key=lambda t: t[1], reverse=True,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    HOLDOUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(HOLDOUT_PARQUET, index=False)

    metadata = {
        "model_name": "WESAD HRV stress classifier (Random Forest)",
        "task": "binary classification: stress vs. no_stress",
        "label_names": LABEL_NAMES,
        "threshold": THRESHOLD,
        "features": FEATURES,
        "dataset": {
            "name": "WESAD HRV features (GeorgiaCh96/WESAD_raw_data)",
            "source": "https://huggingface.co/datasets/GeorgiaCh96/WESAD_raw_data",
            "original": "https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html",
            "rows_after_dedup": int(len(df)),
            "subjects": sorted(int(s) for s in df.subject_id.unique()),
        },
        "split": {
            "test_subjects": TEST_SUBJECTS,
            "train_subjects": sorted(int(s) for s in groups.unique()),
            "train_rows": int(len(train_s)),
            "train_stride": TRAIN_STRIDE,
            "test_rows": int(len(test_df)),
        },
        "hyperparameters": model.named_steps["rf"].get_params(),
        "cv_loso": {"summary": cv_summary, "folds": folds},
        "test_metrics": test_metrics,
        "test_metrics_per_subject": per_subject_test,
        "feature_importances": [{"feature": f, "importance": round(float(i), 4)} for f, i in importances],
        "example_input": {f: float(test_df.iloc[0][f]) for f in FEATURES},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("\nLOSO CV (12 training subjects), mean ± std:")
    for k, v in cv_summary.items():
        print(f"  {k:<10} {v['mean']:.3f} ± {v['std']:.3f}")
    print(f"\nHeld-out test (subjects {TEST_SUBJECTS}, {len(test_df)} windows):")
    for k, v in test_metrics.items():
        print(f"  {k:<16} {v}")
    print(f"\nSaved {MODEL_PATH} and {METADATA_PATH} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
