"""Replay the causal purple pipeline and export fleet/capacity diagnostics."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.filters.smooth_tracking import AISmoothTrackingFilter
from src.core.filters.smooth_tracking.capacity import capacity_for_vehicle, normalize_vehicle_id
from src.core.filters.smooth_tracking.dataframe import filter_smooth_tracking_dataframe
from src.dashboard.dashboard_data import _predict_causal_states_batch, load_telemetry_csv

OUT = ROOT / "artifacts"


def _transitions(values, names):
    return sum(value in names and (i == 0 or values[i-1] != value) for i, value in enumerate(values))


def _band(capacity):
    if capacity is None: return "UNKNOWN"
    if capacity <= 110: return "80-100L"
    if capacity <= 250: return "~200L"
    if capacity <= 450: return "350-400L"
    return "550-800L"


def replay_file(path: Path, predictor: AISmoothTrackingFilter):
    vehicle = normalize_vehicle_id(path.stem.replace("_da_gop", ""))
    capacity = capacity_for_vehicle(vehicle)
    frame = load_telemetry_csv(path)
    engine = AISmoothTrackingFilter(model_dir="")
    pieces = []
    for segment_id, segment in frame.groupby("SegmentID", sort=False, dropna=False):
        segment = segment.sort_values("FuelTime", kind="stable").copy()
        states = _predict_causal_states_batch(segment, predictor, f"{vehicle}:{segment_id}:replay", capacity)
        segment["AI_State"] = states
        segment["AI_Probability"] = states.attrs["probabilities"]
        pieces.append(filter_smooth_tracking_dataframe(segment, vehicle, capacity, engine))
    out = pd.concat(pieces).sort_index(kind="stable")
    raw = pd.to_numeric(out["FuelLevel"], errors="coerce").to_numpy(float)
    clean = pd.to_numeric(out["CleanFuel"], errors="coerce").to_numpy(float)
    states = out["OperationalState"].astype(str).tolist()
    dt_days = max((pd.to_datetime(out["FuelTime"]).max()-pd.to_datetime(out["FuelTime"]).min()).total_seconds()/86400, 1/24)
    raw_steps, clean_steps = np.diff(raw), np.diff(clean)
    correlation = float(np.corrcoef(raw_steps, clean_steps)[0, 1]) if len(raw_steps) > 2 and np.std(raw_steps) and np.std(clean_steps) else np.nan
    confirmation_delays, reacquisition_delays = [], []
    excursion_start = rebound_time = None
    false_shifts = strong_up_count = 0
    times = pd.to_datetime(out["FuelTime"]).tolist()
    for i, state in enumerate(states):
        if bool(out.iloc[i]["ExcursionActive"]) and (i == 0 or not bool(out.iloc[i-1]["ExcursionActive"])):
            excursion_start = times[i]
        if state in ("DOWNWARD_CONFIRMED", "UPWARD_CONFIRMED") and (i == 0 or states[i-1] != state):
            if excursion_start is not None: confirmation_delays.append((times[i]-excursion_start).total_seconds()/60)
            base = out.iloc[i]["ExcursionBaseline"]
            scale = capacity or max(abs(raw[i]), 30)
            if state == "UPWARD_CONFIRMED" and pd.notna(base) and raw[i]-base >= .05*scale:
                strong_up_count += 1
            if pd.notna(base) and any(abs(v-base) <= .01*scale for v in raw[i+1:i+4]): false_shifts += 1
        if state == "U_SHAPE_CONFIRMED": rebound_time = times[i]
        if state == "BASELINE_REACQUISITION" and rebound_time is not None and (i == 0 or states[i-1] != state):
            reacquisition_delays.append((times[i]-rebound_time).total_seconds()/60); rebound_time = None
    row = {
        "VehicleID": vehicle, "CapacityMode": "KNOWN" if capacity else "UNKNOWN", "Capacity": capacity,
        "CapacityBand": _band(capacity), "RecordCount": len(out),
        "ExcursionCount": _transitions(states, {"PENDING_DOWNWARD", "PENDING_UPWARD"}),
        "LongExcursionCount": _transitions(((pd.to_numeric(out["ExcursionElapsedMin"], errors="coerce") >= 20) & out["ExcursionActive"].astype(bool)).tolist(), {True}),
        "ReboundCancelCount": _transitions(states, {"U_SHAPE_CONFIRMED"}),
        "DOWNCandidateCount": _transitions(states, {"PENDING_DOWNWARD"}), "DOWNConfirmedCount": _transitions(states, {"DOWNWARD_CONFIRMED"}),
        "UPCandidateCount": _transitions(states, {"PENDING_UPWARD"}), "UPConfirmedCount": _transitions(states, {"UPWARD_CONFIRMED"}),
        "StrongUPConfirmedCount": strong_up_count,
        "TrendEscapeCount": _transitions(states, {"PERSISTENT_TREND_ESCAPE"}), "GradualTrackingCount": int(sum(s in ("GRADUAL_TRACKING", "PERSISTENT_TREND_ESCAPE") for s in states)),
        "InnovationGatedCount": int(out["InnovationGated"].astype(bool).sum()), "BaselineReacquisitionCount": _transitions(states, {"BASELINE_REACQUISITION"}),
        "MeanAbsStepRaw": float(np.nanmean(abs(raw_steps))), "MeanAbsStepClean": float(np.nanmean(abs(clean_steps))),
        "TotalVariationRaw": float(np.nansum(abs(raw_steps))), "TotalVariationClean": float(np.nansum(abs(clean_steps))),
        "TrendCorrelation": correlation,
        "TrendPreservation": float(abs(clean[-1]-clean[0])/max(abs(raw[-1]-raw[0]), 1e-9)),
        "FalseShiftPerVehicleDay": false_shifts/dt_days,
        "MedianConfirmationDelay": statistics.median(confirmation_delays) if confirmation_delays else np.nan,
        "P90ConfirmationDelay": float(np.percentile(confirmation_delays, 90)) if confirmation_delays else np.nan,
        "MedianReacquisitionDelay": statistics.median(reacquisition_delays) if reacquisition_delays else np.nan,
    }
    return row, out


def main():
    OUT.mkdir(exist_ok=True)
    predictor = AISmoothTrackingFilter(model_dir=str(ROOT / "models/fuel_state_classifier"))
    rows, traces = [], {}
    for path in sorted((ROOT / "fulltt").glob("*.xlsx")):
        row, trace = replay_file(path, predictor); rows.append(row); traces[row["VehicleID"]] = trace
        print(row["VehicleID"], row["RecordCount"], row["TrendEscapeCount"])
    fleet = pd.DataFrame(rows)
    fleet.to_csv(OUT / "operational_guard_fleet_replay.csv", index=False)
    numeric = [column for column in fleet.select_dtypes(include="number").columns if column != "Capacity"]
    summary = fleet.groupby("CapacityBand", dropna=False)[numeric].mean().reindex(["80-100L", "~200L", "350-400L", "550-800L", "UNKNOWN"]).reset_index()
    summary.to_csv(OUT / "operational_guard_capacity_summary.csv", index=False)

    if "29H75028" in traces:
        new = traces["29H75028"].iloc[2192:2213].copy()
        old_path = OUT / "29H75028_2192_2212_operational_guard.csv"
        old = pd.read_csv(old_path)["CleanOld"].tolist() if old_path.exists() else [np.nan]*len(new)
        case = pd.DataFrame({"Index": new.index, "Raw": new.FuelLevel, "CleanOld": old[:len(new)], "CleanNew": new.CleanFuel,
            "ModelState": new.ModelState, "OperationalState": new.OperationalState, "Q": new.KalmanQ, "R": new.KalmanR,
            "rebound_ratio": new.ReboundRatio, "expected_rate": new.ExpectedFuelRate, "observed_rate": new.ObservedFuelRate})
        case.to_csv(old_path, index=False)
    if "29E45520" in traces:
        trace = traces["29E45520"].iloc[265:305]
        trace[["FuelTime", "FuelLevel", "CleanFuel", "ModelState", "OperationalState", "TrendConfidence", "TrendEscapeTriggered", "KalmanQ", "KalmanR"]].to_csv(OUT / "29E45520_persistent_downtrend.csv", index=True)


if __name__ == "__main__":
    main()
