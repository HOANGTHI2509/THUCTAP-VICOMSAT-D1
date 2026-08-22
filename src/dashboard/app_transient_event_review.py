import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\all_labeled_points.csv"
CANDIDATE_PATH = r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\transient_event_candidates.csv"
REVIEW_PATH = r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\transient_event_review.csv"


@st.cache_data
def load_points(path):
    df = pd.read_csv(path)
    df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
    if "SegmentID" not in df.columns:
        df["SegmentID"] = 0
    return df


@st.cache_data
def load_candidates(path):
    df = pd.read_csv(path)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    if "SegmentID" not in df.columns:
        df["SegmentID"] = 0
    return df


@st.cache_data
def load_review(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["event_id", "decision", "final_label", "note"])
    return pd.read_csv(path)


def save_review_row(path, event_id, decision, final_label, note):
    existing = load_review(path)
    existing = existing[existing["event_id"].astype(str) != str(event_id)]
    row = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "decision": decision,
                "final_label": final_label,
                "note": note,
            }
        ]
    )
    updated = pd.concat([existing, row], ignore_index=True)
    updated.to_csv(path, index=False, encoding="utf-8-sig")
    load_review.clear()


def build_event_figure(points, candidate, padding_points):
    vehicle = candidate["VehicleID"]
    segment = candidate.get("SegmentID", 0)
    start_row = int(candidate["start_row"])
    end_row = int(candidate["end_row"])

    group = points[(points["VehicleID"] == vehicle) & (points["SegmentID"].astype(str) == str(segment))].copy()
    if group.empty:
        group = points[points["VehicleID"] == vehicle].copy()
    group = group.sort_values("FuelTime", kind="stable").reset_index(drop=False)

    selected_positions = group.index[(group["index"] >= start_row) & (group["index"] <= end_row)].tolist()
    if selected_positions:
        left = max(0, min(selected_positions) - padding_points)
        right = min(len(group) - 1, max(selected_positions) + padding_points)
    else:
        left = 0
        right = min(len(group) - 1, padding_points * 2)

    view = group.iloc[left : right + 1].copy()
    event = view[(view["index"] >= start_row) & (view["index"] <= end_row)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["FuelTime"],
            y=view["FuelLevel"],
            mode="lines+markers",
            name="Raw FuelLevel",
            line=dict(color="red", width=1.5),
            marker=dict(size=4),
        )
    )
    if not event.empty:
        fig.add_trace(
            go.Scatter(
                x=event["FuelTime"],
                y=event["FuelLevel"],
                mode="markers",
                name="Candidate event",
                marker=dict(color="#8A2BE2", size=10, symbol="diamond"),
            )
        )

    baseline_before = float(candidate["baseline_before"])
    baseline_after = float(candidate["baseline_after"])
    fig.add_hline(y=baseline_before, line=dict(color="#2CA02C", dash="dash"), annotation_text="before")
    fig.add_hline(y=baseline_after, line=dict(color="#1F77B4", dash="dash"), annotation_text="after")
    fig.add_vrect(
        x0=candidate["start_time"],
        x1=candidate["end_time"],
        fillcolor="rgba(138, 43, 226, 0.12)",
        line_width=0,
    )
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
        title="Transient event candidate",
        xaxis_title="Time",
        yaxis_title="FuelLevel (L)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>")
    return fig


def filtered_options_with_default(default_value):
    options = [
        "TRANSIENT_NOISE",
        "TRANSIENT_UP_NOISE",
        "TRANSIENT_DOWN_NOISE",
        "TRANSIENT_CLUSTER_NOISE",
        "NOT_TRANSIENT",
        "REFUEL",
        "DRAIN",
        "SLOSHING_NOISE",
        "SPIKE",
        "UNKNOWN",
    ]
    if default_value not in options:
        options.insert(0, default_value)
    return options


st.set_page_config(page_title="Transient Event Review", layout="wide")
st.title("Transient Event Review")

if not os.path.exists(DATA_PATH) or not os.path.exists(CANDIDATE_PATH):
    st.error("Missing data or candidate CSV. Run build_fuel_label_dataset.py and build_transient_event_candidates.py first.")
    st.stop()

points = load_points(DATA_PATH)
candidates = load_candidates(CANDIDATE_PATH).reset_index(drop=True)
candidates["event_id"] = candidates.index.astype(int)
review = load_review(REVIEW_PATH)
reviewed_ids = set(review["event_id"].astype(str).tolist()) if not review.empty else set()
candidates["Reviewed"] = candidates["event_id"].astype(str).isin(reviewed_ids)

with st.sidebar:
    st.header("Bo loc")
    vehicle_options = ["ALL"] + sorted(candidates["VehicleID"].dropna().astype(str).unique().tolist())
    vehicle = st.selectbox("VehicleID", vehicle_options)
    direction_options = ["ALL"] + sorted(candidates["direction"].dropna().astype(str).unique().tolist())
    direction = st.selectbox("Direction", direction_options)
    min_return = st.slider("Min return_ratio", 0.0, 1.0, 0.55, 0.05)
    max_return = st.slider("Max return_ratio", 0.0, 1.0, 1.0, 0.05)
    min_amp = st.slider("Min amplitude", 0.0, float(max(candidates["amplitude"].max(), 1.0)), 0.0, 0.5)
    show_reviewed = st.checkbox("Hien event da review", value=False)
    padding_points = st.slider("So diem xem quanh event", 10, 160, 45, 5)

filtered = candidates.copy()
if vehicle != "ALL":
    filtered = filtered[filtered["VehicleID"].astype(str) == vehicle]
if direction != "ALL":
    filtered = filtered[filtered["direction"].astype(str) == direction]
filtered = filtered[(filtered["return_ratio"] >= min_return) & (filtered["return_ratio"] <= max_return)]
filtered = filtered[filtered["amplitude"] >= min_amp]
if not show_reviewed:
    filtered = filtered[~filtered["Reviewed"]]
filtered = filtered.sort_values(["return_ratio", "amplitude"], ascending=[False, False]).reset_index(drop=True)

st.caption(f"Candidates: {len(candidates):,} | Filtered: {len(filtered):,} | Reviewed: {len(reviewed_ids):,}")

if filtered.empty:
    st.info("Khong co candidate phu hop bo loc.")
    st.stop()

index = st.number_input("Candidate index trong danh sach loc", min_value=0, max_value=len(filtered) - 1, value=0, step=1)
candidate = filtered.iloc[int(index)]

left, right = st.columns([2, 1])
with right:
    st.subheader("Thong tin event")
    st.dataframe(
        pd.DataFrame([candidate.drop(labels=["Reviewed"]).to_dict()]),
        use_container_width=True,
        hide_index=True,
    )
    decision = st.selectbox("Decision", ["ACCEPT", "REJECT", "REFUEL", "DRAIN", "SLOSHING_LONG", "UNCERTAIN"])
    final_label_default = candidate["suggested_label"] if decision == "ACCEPT" else "NOT_TRANSIENT"
    if final_label_default in {"TRANSIENT_UP_NOISE", "TRANSIENT_DOWN_NOISE", "TRANSIENT_CLUSTER_NOISE"}:
        final_label_default = "TRANSIENT_NOISE"
    final_label_options = filtered_options_with_default(final_label_default)
    final_label = st.selectbox(
        "Final label",
        final_label_options,
        index=final_label_options.index(final_label_default),
    )
    note = st.text_input("Note", "")
    if st.button("Luu review", type="primary"):
        save_review_row(REVIEW_PATH, int(candidate["event_id"]), decision, final_label, note)
        st.success(f"Da luu event_id={int(candidate['event_id'])}")

with left:
    fig = build_event_figure(points, candidate, padding_points)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Danh sach candidate da loc")
st.dataframe(
    filtered[
        [
            "event_id",
            "VehicleID",
            "direction",
            "start_time",
            "end_time",
            "amplitude",
            "duration_points",
            "return_ratio",
            "suggested_label",
            "current_labels",
            "Reviewed",
        ]
    ].head(300),
    use_container_width=True,
    hide_index=True,
)
