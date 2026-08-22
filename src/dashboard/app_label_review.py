import os
import json
import pickle
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATA_PATH = os.path.join("data", "fuel_label_dataset", "all_labeled_points.csv")
MODEL_DIR = os.path.join("models", "fuel_state_classifier")
MODEL_PATH = os.path.join(MODEL_DIR, "fuel_state_classifier.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

LABEL_COLORS = {
    "STABLE_JITTER": "#2CA02C",
    "NORMAL": "#17BECF",
    "CONSUMPTION": "#1F77B4",
    "SLOSHING_NOISE": "#FF7F0E",
    "TRANSIENT_NOISE": "#BCBD22",
    "TRANSIENT_UP_NOISE": "#BCBD22",
    "TRANSIENT_DOWN_NOISE": "#17BECF",
    "TRANSIENT_CLUSTER_NOISE": "#7F7F7F",
    "REFUEL": "#00CC96",
    "DRAIN": "#D62728",
    "SPIKE": "#111111",
    "SPIKE_UP": "#9467BD",
    "SPIKE_DOWN": "#8C564B",
    "DROPOUT": "#E377C2",
    "INVALID": "#7F7F7F",
    "UNKNOWN": "#BDBDBD",
}

IMPORTANT_LABELS = [
    "REFUEL",
    "DRAIN",
    "SPIKE",
    "SPIKE_UP",
    "SPIKE_DOWN",
    "DROPOUT",
    "SLOSHING_NOISE",
    "TRANSIENT_NOISE",
    "TRANSIENT_UP_NOISE",
    "TRANSIENT_DOWN_NOISE",
    "TRANSIENT_CLUSTER_NOISE",
]

AI_MARKER_SYMBOLS = {
    "REFUEL": "triangle-up",
    "DRAIN": "triangle-down",
    "SPIKE": "x",
    "TRANSIENT_NOISE": "diamond-wide",
    "TRANSIENT_UP_NOISE": "diamond-wide",
    "TRANSIENT_DOWN_NOISE": "diamond-wide",
    "TRANSIENT_CLUSTER_NOISE": "diamond-wide",
    "SLOSHING_NOISE": "diamond",
    "CONSUMPTION": "circle-open",
    "STABLE_JITTER": "square-open",
}


@st.cache_data(show_spinner=False)
def load_labeled_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
    df = df[df["FuelTime"].notna()].copy()
    numeric_cols = [
        "FuelLevel",
        "Speed",
        "MotionSpeedKmh",
        "TimeGapMinutes",
        "DeltaFuel",
        "AbsDeltaFuel",
        "RollingStd12",
        "DistanceMeters",
        "GpsSpeedKmh",
        "capacity_est",
        "noise_sigma_liters",
        "flat_jitter_threshold",
        "spike_threshold",
        "event_threshold",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["VehicleID", "FuelTime"], kind="stable")


@st.cache_resource(show_spinner=False)
def load_ai_model(model_path: str, metadata_path: str):
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    return model, metadata


def add_ai_predictions(df: pd.DataFrame, model, metadata: dict | None) -> pd.DataFrame:
    df = df.copy()
    df["AI_Raw_Label"] = ""
    df["AI_Label"] = ""
    df["AI_Confidence"] = pd.NA
    if model is None or not metadata:
        return df

    feature_columns = metadata.get("feature_columns", [])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    x_data = df[feature_columns].values
    df["AI_Raw_Label"] = model.predict(x_data)
    df["AI_Label"] = df["AI_Raw_Label"]
    if hasattr(model, "predict_proba"):
        df["AI_Confidence"] = model.predict_proba(x_data).max(axis=1)

    df = postprocess_ai_labels(df)
    return df


def postprocess_ai_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["VehicleID", "FuelTime"], kind="stable").copy()
    for _, group in df.groupby("VehicleID", sort=False):
        indices = list(group.index)
        for position, index in enumerate(indices):
            if position == 0:
                continue
            row = df.loc[index]
            prev_row = df.loc[indices[position - 1]]
            delta = float(row.get("DeltaFuel", 0.0) or 0.0)
            spike_threshold = float(row.get("spike_threshold", 0.0) or 0.0)
            event_threshold = float(row.get("event_threshold", 0.0) or 0.0)
            flat = float(row.get("flat_jitter_threshold", 0.0) or 0.0)
            fuel = float(row.get("FuelLevel", 0.0) or 0.0)
            prev_fuel = float(prev_row.get("FuelLevel", 0.0) or 0.0)
            next_indices = indices[position + 1 : position + 4]
            if not next_indices:
                continue
            next_fuels = [float(df.loc[next_index, "FuelLevel"]) for next_index in next_indices]
            future_median = float(pd.Series(next_fuels).median())
            drop_is_large = delta <= -max(spike_threshold, event_threshold * 0.7)
            recovers_quickly = abs(future_median - prev_fuel) <= max(flat * 2.0, abs(delta) * 0.30)
            single_point_low = min(next_fuels) > fuel + max(flat, abs(delta) * 0.25)
            if drop_is_large and (recovers_quickly or single_point_low):
                df.loc[index, "AI_Label"] = "SPIKE"
                continue

            if row.get("AI_Label") == "DRAIN":
                next_near_low = sum(
                    1 for next_fuel in next_fuels if abs(next_fuel - fuel) <= max(flat * 2.0, abs(delta) * 0.25)
                )
                if next_near_low < 2:
                    df.loc[index, "AI_Label"] = "SPIKE"
    return df


def add_label_markers(fig: go.Figure, df: pd.DataFrame, selected_labels: list[str]) -> None:
    for label in selected_labels:
        points = df[df["Label"] == label]
        if points.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=points["FuelTime"],
                y=points["FuelLevel"],
                mode="markers",
                name=label,
                marker=dict(size=7, color=LABEL_COLORS.get(label, "#111111"), symbol="circle"),
                text=[
                    f"{label}<br>Fuel={fuel:.2f}<br>Delta={delta:.2f}<br>MotionSpeed={speed:.1f}"
                    for fuel, delta, speed in zip(
                        points["FuelLevel"],
                        points["DeltaFuel"],
                        points["MotionSpeedKmh"] if "MotionSpeedKmh" in points else points["Speed"],
                    )
                ],
                hoverinfo="text+x",
            ),
            row=1,
            col=1,
        )


def add_ai_markers(fig: go.Figure, df: pd.DataFrame, selected_ai_labels: list[str]) -> None:
    for label in selected_ai_labels:
        points = df[df["AI_Label"] == label]
        if points.empty:
            continue
        confidence = (
            points["AI_Confidence"].map(lambda value: "N/A" if pd.isna(value) else f"{value:.2f}")
            if "AI_Confidence" in points
            else pd.Series(["N/A"] * len(points), index=points.index)
        )
        fig.add_trace(
            go.Scattergl(
                x=points["FuelTime"],
                y=points["FuelLevel"],
                mode="markers",
                name=f"AI_{label}",
                marker=dict(
                    size=10,
                    color=LABEL_COLORS.get(label, "#111111"),
                    symbol=AI_MARKER_SYMBOLS.get(label, "x"),
                    line=dict(width=1.5, color="#111111"),
                ),
                text=[
                    f"AI_{ai}<br>Rule={rule}<br>Fuel={fuel:.2f}<br>Delta={delta:.2f}<br>Conf={conf}"
                    for ai, rule, fuel, delta, conf in zip(
                        points["AI_Label"],
                        points["Label"],
                        points["FuelLevel"],
                        points["DeltaFuel"],
                        confidence,
                    )
                ],
                hoverinfo="text+x",
            ),
            row=1,
            col=1,
        )


st.set_page_config(page_title="Fuel Label Review", layout="wide")
st.title("Fuel Label Review")

if not os.path.exists(DATA_PATH):
    st.error(f"Không tìm thấy dataset: {DATA_PATH}")
    st.stop()

df_all = load_labeled_data(DATA_PATH)
ai_model = None
ai_metadata = None

with st.sidebar:
    st.header("Bộ lọc")
    vehicles = sorted(df_all["VehicleID"].dropna().unique().tolist())
    vehicle = st.selectbox("VehicleID", vehicles)

    df_vehicle = df_all[df_all["VehicleID"] == vehicle].copy()
    if df_vehicle.empty:
        st.warning("Không có dữ liệu thời gian hợp lệ cho xe này.")
        st.stop()
    min_time = df_vehicle["FuelTime"].min()
    max_time = df_vehicle["FuelTime"].max()
    if pd.isna(min_time) or pd.isna(max_time):
        st.warning("Không có FuelTime hợp lệ cho xe này.")
        st.stop()
    date_range = st.date_input(
        "Khoảng ngày",
        value=(min_time.date(), max_time.date()),
        min_value=min_time.date(),
        max_value=max_time.date(),
    )

    labels = sorted(df_vehicle["Label"].dropna().unique().tolist())
    default_labels = [label for label in IMPORTANT_LABELS if label in labels]
    selected_labels = st.multiselect("Nhãn cần tô màu", labels, default=default_labels)
    enable_ai = st.checkbox("Bật AI prediction", value=False)
    ai_labels = ["STABLE_JITTER", "CONSUMPTION", "SLOSHING_NOISE", "TRANSIENT_NOISE", "REFUEL", "DRAIN", "SPIKE"] if enable_ai else []
    default_ai_labels = []
    selected_ai_labels = st.multiselect("AI nhãn cần tô màu", ai_labels, default=default_ai_labels)

    show_unknown = st.checkbox("Hiện UNKNOWN", value=False)
    show_stable = st.checkbox("Hiện STABLE/NORMAL", value=False)
    show_ai_disagreements = st.checkbox("Chỉ hiện Rule != AI trong bảng", value=True)
    max_points = st.slider("Giới hạn điểm vẽ", 1000, 80000, 30000, step=1000)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
df = df_vehicle[(df_vehicle["FuelTime"] >= start_ts) & (df_vehicle["FuelTime"] < end_ts)].copy()
if enable_ai:
    ai_model, ai_metadata = load_ai_model(MODEL_PATH, METADATA_PATH)
    df = add_ai_predictions(df, ai_model, ai_metadata)
else:
    df["AI_Raw_Label"] = ""
    df["AI_Label"] = ""
    df["AI_Confidence"] = pd.NA

if not show_unknown:
    selected_labels = [label for label in selected_labels if label != "UNKNOWN"]
if show_unknown and "UNKNOWN" not in selected_labels and "UNKNOWN" in labels:
    selected_labels.append("UNKNOWN")
if show_stable:
    for label in ["STABLE_JITTER", "NORMAL", "CONSUMPTION"]:
        if label in labels and label not in selected_labels:
            selected_labels.append(label)

if len(df) > max_points:
    step = max(1, len(df) // max_points)
    sampled_df = df.iloc[::step].copy()
else:
    sampled_df = df.copy()

selected_marker_df = df[df["Label"].isin(selected_labels)].copy()
selected_ai_marker_df = df[df["AI_Label"].isin(selected_ai_labels)].copy()
if len(selected_ai_marker_df) > max_points:
    selected_ai_marker_df = selected_ai_marker_df.iloc[:: max(1, len(selected_ai_marker_df) // max_points)].copy()
df_plot = (
    pd.concat([sampled_df, selected_marker_df, selected_ai_marker_df], axis=0)
    .sort_values("FuelTime", kind="stable")
    .loc[lambda data: ~data.index.duplicated(keep="first")]
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Số bản ghi", f"{len(df):,}")
col2.metric("GPS hợp lệ", f"{int(df['HasGPS'].sum()):,}" if "HasGPS" in df else "N/A")
col3.metric("Fuel min/max", f"{df['FuelLevel'].min():.1f} / {df['FuelLevel'].max():.1f}")
col4.metric("UNKNOWN", f"{int((df['Label'] == 'UNKNOWN').sum()):,}")
col5.metric("Rule != AI", f"{int((df['Label'] != df['AI_Label']).sum()):,}" if enable_ai else "N/A")

label_counts = df["Label"].value_counts().rename_axis("Label").reset_index(name="Count")
if enable_ai:
    ai_counts = df["AI_Label"].value_counts().rename_axis("AI_Label").reset_index(name="Count")
    left_counts, right_counts = st.columns(2)
    left_counts.dataframe(label_counts, use_container_width=True, hide_index=True)
    right_counts.dataframe(ai_counts, use_container_width=True, hide_index=True)
else:
    st.dataframe(label_counts, use_container_width=True, hide_index=True)

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.62, 0.2, 0.18],
    subplot_titles=("FuelLevel + nhãn", "MotionSpeed", "RollingStd12 / AbsDeltaFuel"),
)

fig.add_trace(
    go.Scattergl(
        x=df_plot["FuelTime"],
        y=df_plot["FuelLevel"],
        mode="lines+markers",
        name="Raw FuelLevel",
        line=dict(color="#D62728", width=1.2),
        marker=dict(size=3),
    ),
    row=1,
    col=1,
)

add_label_markers(fig, df_plot, selected_labels)
add_ai_markers(fig, df_plot, selected_ai_labels)

fig.add_trace(
    go.Scattergl(
        x=df_plot["FuelTime"],
        y=df_plot["MotionSpeedKmh"] if "MotionSpeedKmh" in df_plot else df_plot["Speed"],
        mode="lines",
        name="MotionSpeed",
        line=dict(color="#636EFA", width=1.2),
        fill="tozeroy",
        fillcolor="rgba(99, 110, 250, 0.12)",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scattergl(
        x=df_plot["FuelTime"],
        y=df_plot["RollingStd12"],
        mode="lines",
        name="RollingStd12",
        line=dict(color="#FF7F0E", width=1.2),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scattergl(
        x=df_plot["FuelTime"],
        y=df_plot["AbsDeltaFuel"],
        mode="lines",
        name="AbsDeltaFuel",
        line=dict(color="#9467BD", width=1.0),
        visible="legendonly",
    ),
    row=3,
    col=1,
)

fig.update_layout(
    height=850,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=80, b=30),
)
fig.update_yaxes(title_text="Lít", row=1, col=1)
fig.update_yaxes(title_text="km/h", row=2, col=1)
fig.update_yaxes(title_text="Lít", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

if enable_ai and ai_model is None:
    st.warning(f"Chưa thấy model AI: {MODEL_PATH}")
elif enable_ai:
    st.subheader("Rule Label vs AI Label")
    compare_df = df.copy()
    if show_ai_disagreements:
        compare_df = compare_df[compare_df["Label"] != compare_df["AI_Label"]]
    st.dataframe(
        compare_df[
            [
                "VehicleID",
                "FuelTime",
                "FuelLevel",
                "DeltaFuel",
                "RollingStd12",
                "MotionSpeedKmh",
                "Label",
                "AI_Raw_Label",
                "AI_Label",
                "AI_Confidence",
            ]
        ].head(1000),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Các điểm nhãn quan trọng")
important_df = df[df["Label"].isin(IMPORTANT_LABELS)].copy()
if important_df.empty:
    st.info("Không có điểm nhãn quan trọng trong khoảng đang chọn.")
else:
    st.dataframe(
        important_df[
            [
                "VehicleID",
                "FuelTime",
                "FuelLevel",
                "DeltaFuel",
                "RollingStd12",
                "Speed",
                "MotionSpeedKmh",
                "GpsSpeedKmh",
                "DistanceMeters",
                "Label",
            ]
        ].head(1000),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Mẫu UNKNOWN biên lớn")
unknown_large = df[(df["Label"] == "UNKNOWN") & (df["AbsDeltaFuel"] >= df["spike_threshold"])].copy()
if unknown_large.empty:
    st.info("Không có UNKNOWN delta lớn trong khoảng đang chọn.")
else:
    st.dataframe(
        unknown_large[
            [
                "VehicleID",
                "FuelTime",
                "FuelLevel",
                "DeltaFuel",
                "AbsDeltaFuel",
                "RollingStd12",
                "Speed",
                "MotionSpeedKmh",
                "GpsSpeedKmh",
                "Label",
            ]
        ].head(1000),
        use_container_width=True,
        hide_index=True,
    )
