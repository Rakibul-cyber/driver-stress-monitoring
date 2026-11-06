"""Model loading, single-sample prediction and SIMULATED real-time replay.

The replay streams recorded HRV feature windows from held-out WESAD subjects
one at a time. It is a simulation of streaming inference, not a live
vehicle sensor integration.

Usage:  python -m src.inference --interval 0.2 --limit 30 --subject 16
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import (
    FEATURES,
    HOLDOUT_PARQUET,
    LABEL_NAMES,
    METADATA_PATH,
    MODEL_PATH,
    features_to_frame,
)


class StressPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found. Run `python -m src.train` first.")
        self.model = joblib.load(model_path)
        # Single-row predictions are faster without the parallel thread pool.
        self.model.named_steps["rf"].n_jobs = 1
        self.metadata = json.loads(Path(metadata_path).read_text())
        self.threshold = float(self.metadata["threshold"])
        if self.metadata["features"] != FEATURES:
            raise ValueError("Model feature list does not match src.preprocessing.FEATURES")

    def predict(self, features: dict[str, float]) -> dict:
        """Predict a single window. Raises ValueError on invalid input."""
        start = time.perf_counter()
        X = features_to_frame(features)
        p_stress = float(self.model.predict_proba(X)[0, 1])
        label_id = int(p_stress >= self.threshold)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "label": LABEL_NAMES[label_id],
            "label_id": label_id,
            "probabilities": {"no_stress": round(1 - p_stress, 4), "stress": round(p_stress, 4)},
            "confidence": round(max(p_stress, 1 - p_stress), 4),
            "latency_ms": round(latency_ms, 3),
        }


def load_holdout(path: Path = HOLDOUT_PARQUET) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    return pd.read_parquet(path)


def replay(
    predictor: StressPredictor,
    data: pd.DataFrame,
    interval_s: float = 1.0,
    step: int = 4,
    start: int = 0,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield predictions for recorded windows in time order (simulated stream).

    step: rows to advance per tick (source windows are spaced 0.25 s apart,
          so step=4 replays one window per second of recording).
    """
    rows = data.iloc[start::step]
    if limit is not None:
        rows = rows.iloc[:limit]
    for i, (_, row) in enumerate(rows.iterrows()):
        if i > 0 and interval_s > 0:
            time.sleep(interval_s)
        features = {f: float(row[f]) for f in FEATURES}
        result = predictor.predict(features)
        yield {
            "subject_id": int(row["subject_id"]),
            "time_min": round(float(row["Time"]), 3),
            "true_condition": row["condition"],
            "true_label": LABEL_NAMES[int(row["label"])],
            "features": features,
            **result,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulated real-time stress inference replay")
    ap.add_argument("--subject", type=int, default=None, help="held-out subject id (default: all)")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between samples")
    ap.add_argument("--step", type=int, default=4, help="rows advanced per tick")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    predictor = StressPredictor()
    data = load_holdout()
    if args.subject is not None:
        data = data[data.subject_id == args.subject]
    print("SIMULATED real-time inference (replay of held-out WESAD recordings)")
    correct, latencies = 0, []
    for n, r in enumerate(replay(predictor, data, args.interval, args.step, args.start, args.limit), 1):
        correct += r["label"] == r["true_label"]
        latencies.append(r["latency_ms"])
        print(f"[S{r['subject_id']} t={r['time_min']:7.2f}min] HR={r['features']['HR']:6.1f} "
              f"pred={r['label']:<9} conf={r['confidence']:.2f} "
              f"true={r['true_condition']:<10} latency={r['latency_ms']:.1f}ms")
    if latencies:
        print(f"\n{n} samples | accuracy {correct / n:.3f} | "
              f"mean latency {sum(latencies) / n:.1f} ms | max {max(latencies):.1f} ms")


if __name__ == "__main__":
    main()
