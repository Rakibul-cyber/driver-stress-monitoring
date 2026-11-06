"""Streamlit dashboard: simulated real-time driver stress monitoring.

Run:  streamlit run app.py
If API_URL is set (e.g. http://api:8000 in Docker Compose), predictions are
requested from the FastAPI service; otherwise the model is loaded in-process.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from src.inference import StressPredictor, load_holdout
from src.preprocessing import FEATURES, LABEL_NAMES

API_URL = os.environ.get("API_URL", "").rstrip("/")
KEY_FEATURES = ["HR", "MEAN_RR", "RMSSD", "SDRR", "pNN50", "LF_HF"]
HISTORY_LEN = 600

st.set_page_config(page_title="Driver Stress Monitoring", page_icon="🚗", layout="wide")


@st.cache_resource
def get_predictor() -> StressPredictor:
    return StressPredictor()


@st.cache_data
def get_holdout() -> pd.DataFrame:
    return load_holdout()


def predict(features: dict[str, float]) -> dict:
    if API_URL:
        r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=5)
        r.raise_for_status()
        return r.json()
    return get_predictor().predict(features)


def reset_stream() -> None:
    st.session_state.pos = None  # resolved to the chosen start minute below
    st.session_state.history = []
    st.session_state.running = False


try:
    predictor = get_predictor()
    holdout = get_holdout()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

meta = predictor.metadata
for k, v in {"pos": 0, "history": [], "running": False}.items():
    st.session_state.setdefault(k, v)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Replay controls")
subjects = sorted(holdout.subject_id.unique().tolist())
subject = st.sidebar.selectbox(
    "Held-out subject", subjects, format_func=lambda s: f"S{s}", on_change=reset_stream
)
interval = st.sidebar.slider("Update interval (s)", 0.1, 2.0, 0.5, 0.1)
step = st.sidebar.slider(
    "Recording seconds per update", 1, 30, 5,
    help="Source windows are 0.25 s apart; larger values fast-forward through the recording.",
)
stream = holdout[holdout.subject_id == subject].reset_index(drop=True)
phases = stream.groupby("condition")["Time"].agg(["min", "max"]).sort_values("min")
st.sidebar.number_input(
    "Start at recording minute", 0.0, float(stream.Time.max()), 0.0, 1.0,
    key="start_min", on_change=reset_stream,
    help="Jump ahead, e.g. to the stress phase listed below.",
)
if st.session_state.pos is None:
    st.session_state.pos = int(stream.Time.searchsorted(st.session_state.start_min))
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Start", width="stretch"):
    st.session_state.running = True
if c2.button("⏸ Stop", width="stretch"):
    st.session_state.running = False
if c3.button("↺ Reset", width="stretch"):
    reset_stream()
    st.session_state.pos = int(stream.Time.searchsorted(st.session_state.start_min))

pos_now = min(st.session_state.pos, len(stream) - 1)
st.sidebar.progress(min(st.session_state.pos / len(stream), 1.0),
                    text=f"t = {stream.Time.iloc[pos_now]:.1f} / {stream.Time.max():.1f} min")
st.sidebar.markdown("**Protocol phases (recording minutes)**")
st.sidebar.dataframe(phases.round(1), width="stretch")
st.sidebar.caption(f"Backend: {'FastAPI @ ' + API_URL if API_URL else 'in-process model'}")

# ---------------- Header ----------------
st.title("🚗 Real-Time Driver Stress & Cognitive Load Monitoring")
st.caption(
    "**Simulated real-time inference**: replays recorded HRV feature windows from WESAD "
    "subjects the model never saw during training. No live vehicle sensors are connected."
)

tab_live, tab_eval = st.tabs(["Live monitor (simulated)", "Model evaluation"])


def advance() -> None:
    pos = st.session_state.pos
    if pos >= len(stream):
        st.session_state.running = False
        return
    row = stream.iloc[pos]
    features = {f: float(row[f]) for f in FEATURES}
    res = predict(features)
    st.session_state.history.append({
        "time_min": float(row["Time"]),
        "p_stress": res["probabilities"]["stress"],
        "pred": res["label"],
        "true": LABEL_NAMES[int(row["label"])],
        "condition": row["condition"],
        "confidence": res["confidence"],
        "latency_ms": res["latency_ms"],
        **{f: features[f] for f in KEY_FEATURES},
        "_features": features,
    })
    st.session_state.history = st.session_state.history[-HISTORY_LEN:]
    st.session_state.pos = pos + step * 4


with tab_live:
    @st.fragment(run_every=interval if st.session_state.running else None)
    def live_panel() -> None:
        if st.session_state.running:
            try:
                advance()
            except requests.RequestException as e:
                st.session_state.running = False
                st.error(f"API request failed: {e}")

        hist = st.session_state.history
        if not hist:
            st.info("Press **▶ Start** in the sidebar to begin replaying the selected subject.")
            return
        cur = hist[-1]
        df = pd.DataFrame(hist).drop(columns="_features")
        is_stress = cur["pred"] == "stress"

        if is_stress:
            st.error(f"⚠️ **STRESS DETECTED** at t = {cur['time_min']:.2f} min "
                     f"(P(stress) = {cur['p_stress']:.2f})")
        else:
            st.success(f"✅ **No stress detected** at t = {cur['time_min']:.2f} min "
                       f"(P(stress) = {cur['p_stress']:.2f})")

        m = st.columns(5)
        m[0].metric("Prediction", "Stress" if is_stress else "No stress")
        m[1].metric("Confidence", f"{cur['confidence']:.0%}")
        m[2].metric("Ground truth", cur["condition"])
        m[3].metric("Running accuracy", f"{(df.pred == df.true).mean():.0%}",
                    help=f"Over the last {len(df)} predictions shown")
        m[4].metric("Inference latency", f"{cur['latency_ms']:.1f} ms")

        left, right = st.columns(2)
        with left:
            st.subheader("Stress probability")
            chart = df.set_index("time_min")[["p_stress"]].assign(
                true_stress=(df.true == "stress").astype(float).values,
                threshold=meta["threshold"],
            )
            st.line_chart(chart, x_label="recording time (min)", height=280)
        with right:
            st.subheader("Heart rate & HRV")
            st.line_chart(df.set_index("time_min")[["HR", "RMSSD"]],
                          x_label="recording time (min)", height=280)

        left, right = st.columns([1, 2])
        with left:
            st.subheader("Incoming features")
            feats = pd.Series(cur["_features"], name="value").round(3)
            st.dataframe(feats.loc[KEY_FEATURES], width="stretch")
            with st.expander("All 30 features"):
                st.dataframe(feats, width="stretch", height=300)
        with right:
            st.subheader("Prediction history")
            st.dataframe(
                df[["time_min", "pred", "p_stress", "condition", "latency_ms"]]
                .iloc[::-1].head(50).round(3),
                width="stretch", hide_index=True, height=380,
            )

    live_panel()


with tab_eval:
    st.markdown(
        f"**Model:** Random Forest ({meta['hyperparameters']['n_estimators']} trees) on "
        f"{len(FEATURES)} HRV features · **Train subjects:** "
        f"{', '.join('S' + str(s) for s in meta['split']['train_subjects'])} · "
        f"**Held-out test subjects:** "
        f"{', '.join('S' + str(s) for s in meta['split']['test_subjects'])}"
    )
    tm = meta["test_metrics"]
    st.subheader("Held-out test subjects")
    cols = st.columns(5)
    for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(k.replace("_", " ").upper() if k == "roc_auc" else k.capitalize(), f"{tm[k]:.3f}")
    st.caption("Precision / recall / F1 are for the *stress* class. "
               f"Evaluated on {meta['split']['test_rows']:,} windows.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix (held-out test)**")
        cm = pd.DataFrame(tm["confusion_matrix"],
                          index=["true: no_stress", "true: stress"],
                          columns=["pred: no_stress", "pred: stress"])
        st.dataframe(cm, width="stretch")
        st.markdown("**Leave-one-subject-out CV (12 training subjects)**")
        cv = pd.DataFrame(meta["cv_loso"]["summary"]).T
        st.dataframe(cv, width="stretch")
    with right:
        st.markdown("**Top 10 feature importances**")
        fi = pd.DataFrame(meta["feature_importances"][:10]).set_index("feature")
        st.bar_chart(fi, horizontal=True, height=320, sort="-importance")

    st.markdown("**Per-subject LOSO results** (large variation across people)")
    folds = pd.DataFrame(meta["cv_loso"]["folds"]).set_index("subject_id")
    st.dataframe(folds[["accuracy", "precision", "recall", "f1", "roc_auc"]],
                 width="stretch")
