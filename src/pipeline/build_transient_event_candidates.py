import argparse
import csv
import os

import numpy as np
import pandas as pd


def _numeric(df, column, default=0.0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _median(values):
    values = [float(value) for value in values if not pd.isna(value)]
    if not values:
        return np.nan
    return float(np.median(values))


def _detect_group_candidates(group, max_duration_points=6):
    group = group.sort_values("FuelTime", kind="stable").reset_index(drop=False)
    fuel = _numeric(group, "FuelLevel", np.nan).to_numpy(dtype=float)
    flat = _numeric(group, "flat_jitter_threshold", 0.8).to_numpy(dtype=float)
    spike = _numeric(group, "spike_threshold", 3.0).to_numpy(dtype=float)
    event = _numeric(group, "event_threshold", 6.0).to_numpy(dtype=float)
    time_values = pd.to_datetime(group["FuelTime"], errors="coerce")
    rows = []

    n = len(group)
    for start in range(3, max(3, n - 3)):
        if pd.isna(fuel[start]):
            continue

        baseline_before = _median(fuel[max(0, start - 4) : start])
        if pd.isna(baseline_before):
            continue

        threshold = max(float(event[start]) * 0.45, float(spike[start]) * 1.10, float(flat[start]) * 4.0, 2.0)
        for end in range(start + 1, min(n - 1, start + max_duration_points) + 1):
            segment = fuel[start : end + 1]
            if np.isnan(segment).any():
                continue

            after = fuel[end + 1 : min(n, end + 5)]
            if len(after) == 0 or np.isnan(after).all():
                continue
            baseline_after = _median(after)
            if pd.isna(baseline_after):
                continue

            peak = float(np.nanmax(segment))
            trough = float(np.nanmin(segment))
            up_amplitude = peak - baseline_before
            down_amplitude = baseline_before - trough

            if up_amplitude >= down_amplitude:
                direction = "UP"
                extreme = peak
                amplitude = up_amplitude
                extreme_offset = int(np.nanargmax(segment))
            else:
                direction = "DOWN"
                extreme = trough
                amplitude = down_amplitude
                extreme_offset = int(np.nanargmin(segment))

            if amplitude < threshold:
                continue

            return_to_before = abs(baseline_after - baseline_before)
            return_gate = max(float(flat[start]) * 2.5, amplitude * 0.30, 1.0)
            if return_to_before > return_gate:
                continue

            duration_points = end - start + 1
            start_time = time_values.iloc[start]
            end_time = time_values.iloc[end]
            duration_minutes = (
                (end_time - start_time).total_seconds() / 60.0
                if pd.notna(start_time) and pd.notna(end_time)
                else np.nan
            )
            return_ratio = 1.0 - return_to_before / max(amplitude, 1e-6)

            # Require an actual excursion inside the candidate, not only a slow ramp.
            if duration_points >= 3:
                diffs = np.diff(segment)
                has_reversal = np.any(diffs > float(flat[start])) and np.any(diffs < -float(flat[start]))
            else:
                has_reversal = True
            if not has_reversal:
                continue

            source_indices = group.loc[start : end, "index"].tolist()
            rows.append(
                {
                    "VehicleID": group.loc[start, "VehicleID"],
                    "SegmentID": group.loc[start, "SegmentID"] if "SegmentID" in group.columns else 0,
                    "Source": group.loc[start, "Source"] if "Source" in group.columns else "",
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_row": int(group.loc[start, "index"]),
                    "end_row": int(group.loc[end, "index"]),
                    "extreme_row": int(group.loc[start + extreme_offset, "index"]),
                    "direction": direction,
                    "baseline_before": baseline_before,
                    "baseline_after": baseline_after,
                    "extreme_value": extreme,
                    "amplitude": amplitude,
                    "duration_points": duration_points,
                    "duration_minutes": duration_minutes,
                    "return_ratio": return_ratio,
                    "suggested_label": f"TRANSIENT_{direction}_NOISE",
                    "current_labels": ",".join(group.loc[start : end, "Label"].astype(str).unique()) if "Label" in group.columns else "",
                    "source_rows": ";".join(str(value) for value in source_indices),
                }
            )
            break

    return rows


def build_candidates(input_path, output_path, max_duration_points=6):
    df = pd.read_csv(input_path)
    df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
    if "SegmentID" not in df.columns:
        df["SegmentID"] = 0
    rows = []
    for _, group in df.groupby(["VehicleID", "SegmentID"], sort=False, dropna=False):
        rows.extend(_detect_group_candidates(group, max_duration_points=max_duration_points))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "VehicleID",
        "SegmentID",
        "Source",
        "start_time",
        "end_time",
        "start_row",
        "end_row",
        "extreme_row",
        "direction",
        "baseline_before",
        "baseline_after",
        "extreme_value",
        "amplitude",
        "duration_points",
        "duration_minutes",
        "return_ratio",
        "suggested_label",
        "current_labels",
        "source_rows",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved transient candidates: {output_path}")
    print(f"Candidate count: {len(rows):,}")


def main():
    parser = argparse.ArgumentParser(description="Build transient-noise event candidates for review.")
    parser.add_argument("--input", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\all_labeled_points.csv")
    parser.add_argument("--output", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\transient_event_candidates.csv")
    parser.add_argument("--max-duration-points", type=int, default=6)
    args = parser.parse_args()
    build_candidates(args.input, args.output, max_duration_points=args.max_duration_points)


if __name__ == "__main__":
    main()
