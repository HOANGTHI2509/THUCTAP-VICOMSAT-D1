import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.filters.ai_state_filter import filter_with_ai_state, load_fuel_state_classifier
from src.core.filters.tcn_state_classifier import load_fuel_state_tcn


def _load_adaptive_kalman_class():
    path = os.path.join(PROJECT_ROOT, "src", "core", "filters", "kalman_adaptive copy.py")
    spec = importlib.util.spec_from_file_location("kalman_adaptive_copy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BoLocKalmanThichNghi1D


def _numeric(df, column, default=0.0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def run_adaptive_kalman(df):
    BoLocKalmanThichNghi1D = _load_adaptive_kalman_class()
    result = df.copy()
    result["AdaptiveKalman_Compare"] = np.nan

    for _, group in result.groupby(["VehicleID", "SegmentID"], sort=False, dropna=False):
        kf = None
        for idx, row in group.sort_values("FuelTime", kind="stable").iterrows():
            raw = pd.to_numeric(row.get("FuelLevel"), errors="coerce")
            if pd.isna(raw) or raw <= 0:
                continue
            if kf is None:
                kf = BoLocKalmanThichNghi1D(
                    trang_thai_ban_dau=float(raw),
                    sai_so_uoc_luong_ban_dau=9.0,
                    nhieu_qua_trinh=1.0,
                    r_co_ban=9.0,
                    r_nhieu_dot_bien=49.0,
                    nguong_bat_nhay=float(row.get("event_threshold", 10.0) or 10.0),
                    nhip_cho_xac_nhan=3,
                )
                filtered = float(raw)
            else:
                speed = float(row.get("Speed", row.get("MotionSpeedKmh", 0.0)) or 0.0)
                motion = 0 if speed <= 3.0 else 1
                gap = float(row.get("TimeGapMinutes", 5.0) or 5.0)
                filtered = kf.cap_nhat(
                    float(raw),
                    ty_le_dt=max(gap / 5.0, 0.1),
                    trang_thai_chuyen_dong=motion,
                    gia_toc=float(row.get("Acceleration", 0.0) or 0.0),
                )
            result.loc[idx, "AdaptiveKalman_Compare"] = filtered
    return result


def run_ai_filter(df, rf_dir, tcn_dir):
    rf_model, rf_metadata = load_fuel_state_classifier(rf_dir)
    tcn_model, tcn_metadata = load_fuel_state_tcn(tcn_dir)
    if rf_model is None or rf_metadata is None:
        raise SystemExit(f"RF model not found: {rf_dir}")
    result = (
        df.groupby(["VehicleID", "SegmentID"], sort=False, dropna=False, group_keys=False)
        .apply(lambda group: filter_with_ai_state(group, rf_model, rf_metadata, tcn_model, tcn_metadata))
        .copy()
    )
    return result


def _mean_abs(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.abs().mean())


def compute_metrics(df, output_col):
    df = df.copy()
    df["EvalLabel"] = df["Label"].replace(
        {
            "TRANSIENT_NOISE": "SLOSHING_NOISE",
            "TRANSIENT_UP_NOISE": "SLOSHING_NOISE",
            "TRANSIENT_DOWN_NOISE": "SLOSHING_NOISE",
            "TRANSIENT_CLUSTER_NOISE": "SLOSHING_NOISE",
            "SPIKE": "SLOSHING_NOISE",
            "SPIKE_UP": "SLOSHING_NOISE",
            "SPIKE_DOWN": "SLOSHING_NOISE",
            "DROPOUT": "SLOSHING_NOISE",
        }
    )
    rows = []
    labels = [
        "STABLE_JITTER",
        "CONSUMPTION",
        "SLOSHING_NOISE",
        "REFUEL",
        "DRAIN",
    ]
    for label in labels:
        part = df[df["EvalLabel"] == label].copy()
        if part.empty:
            continue

        raw = _numeric(part, "FuelLevel", np.nan)
        output = _numeric(part, output_col, np.nan)
        raw_step = raw.groupby([part["VehicleID"], part["SegmentID"]], sort=False).diff()
        output_step = output.groupby([part["VehicleID"], part["SegmentID"]], sort=False).diff()
        residual = output - raw
        raw_step_mae = _mean_abs(raw_step)
        output_step_mae = _mean_abs(output_step)
        movement_ratio = output_step_mae / raw_step_mae if raw_step_mae and not pd.isna(raw_step_mae) else np.nan

        rows.append(
            {
                "Label": label,
                "Count": int(len(part)),
                "MeanAbsOutputStep": output_step_mae,
                "MeanAbsResidualToRaw": _mean_abs(residual),
                "MeanSignedResidual": float(residual.dropna().mean()) if not residual.dropna().empty else np.nan,
                "MovementRatioVsRaw": movement_ratio,
            }
        )
    return rows


def write_comparison(out_dir, adaptive_rows, ai_rows):
    os.makedirs(out_dir, exist_ok=True)
    by_label = {row["Label"]: row for row in adaptive_rows}
    ai_by_label = {row["Label"]: row for row in ai_rows}
    labels = [label for label in by_label.keys() if label in ai_by_label]

    csv_path = os.path.join(out_dir, "adaptive_vs_ai_filter_metrics.csv")
    md_path = os.path.join(out_dir, "adaptive_vs_ai_filter_metrics.md")

    header = [
        "Label",
        "Count",
        "Adaptive MeanAbsOutputStep",
        "AI MeanAbsOutputStep",
        "Adaptive MeanAbsResidualToRaw",
        "AI MeanAbsResidualToRaw",
        "Adaptive MeanSignedResidual",
        "AI MeanSignedResidual",
        "Adaptive MovementRatioVsRaw",
        "AI MovementRatioVsRaw",
    ]

    def fmt(value):
        return "" if pd.isna(value) else f"{float(value):.4f}"

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for label in labels:
            adaptive = by_label[label]
            ai = ai_by_label[label]
            writer.writerow(
                [
                    label,
                    adaptive["Count"],
                    fmt(adaptive["MeanAbsOutputStep"]),
                    fmt(ai["MeanAbsOutputStep"]),
                    fmt(adaptive["MeanAbsResidualToRaw"]),
                    fmt(ai["MeanAbsResidualToRaw"]),
                    fmt(adaptive["MeanSignedResidual"]),
                    fmt(ai["MeanSignedResidual"]),
                    fmt(adaptive["MovementRatioVsRaw"]),
                    fmt(ai["MovementRatioVsRaw"]),
                ]
            )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Adaptive Kalman vs AI Ensemble Filter\n\n")
        handle.write("| Label | Count | Adaptive step | AI step | Adaptive abs residual | AI abs residual | Adaptive bias | AI bias | Adaptive move ratio | AI move ratio |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for label in labels:
            adaptive = by_label[label]
            ai = ai_by_label[label]
            handle.write(
                f"| {label} | {adaptive['Count']} | {fmt(adaptive['MeanAbsOutputStep'])} | {fmt(ai['MeanAbsOutputStep'])} | "
                f"{fmt(adaptive['MeanAbsResidualToRaw'])} | {fmt(ai['MeanAbsResidualToRaw'])} | "
                f"{fmt(adaptive['MeanSignedResidual'])} | {fmt(ai['MeanSignedResidual'])} | "
                f"{fmt(adaptive['MovementRatioVsRaw'])} | {fmt(ai['MovementRatioVsRaw'])} |\n"
            )
        handle.write("\n")
        handle.write("Notes:\n")
        handle.write("- MeanAbsOutputStep: lower is smoother, especially for STABLE_JITTER/SLOSHING_NOISE/SPIKE.\n")
        handle.write("- MeanAbsResidualToRaw: lower means closer to sensor, useful for REFUEL/DRAIN/CONSUMPTION.\n")
        handle.write("- MeanSignedResidual: positive means output tends to stay above raw, often lagging on drops.\n")
        handle.write("- MovementRatioVsRaw: output movement divided by raw movement; lower means stronger denoising.\n")

    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Compare Adaptive Kalman and AI ensemble filter behavior.")
    parser.add_argument("--data", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\test.csv")
    parser.add_argument("--rf-dir", default=r"D:\THUCTAP_VICOMSAT\models\fuel_state_classifier")
    parser.add_argument("--tcn-dir", default=r"D:\THUCTAP_VICOMSAT\models\fuel_state_tcn")
    parser.add_argument("--out-dir", default=r"D:\THUCTAP_VICOMSAT\models\filter_comparison")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
    if "SegmentID" not in df.columns:
        df["SegmentID"] = 0
    df = df.sort_values(["VehicleID", "SegmentID", "FuelTime"], kind="stable")
    df = run_adaptive_kalman(df)
    df = run_ai_filter(df, args.rf_dir, args.tcn_dir)

    adaptive_rows = compute_metrics(df, "AdaptiveKalman_Compare")
    ai_rows = compute_metrics(df, "AI_State_Filtered")
    csv_path, md_path = write_comparison(args.out_dir, adaptive_rows, ai_rows)
    sample_path = os.path.join(args.out_dir, "test_with_filter_outputs.csv")
    df.to_csv(sample_path, index=False, encoding="utf-8-sig")

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison Markdown: {md_path}")
    print(f"Saved row-level outputs: {sample_path}")


if __name__ == "__main__":
    main()
