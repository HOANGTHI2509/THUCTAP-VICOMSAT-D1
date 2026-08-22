import argparse
import csv
import math
import os
from datetime import datetime


def _as_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        number = float(str(value).replace(",", "."))
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _parse_time(value):
    text = str(value or "")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            pass
    return None


def _median(values):
    nums = sorted(value for value in values if value is not None)
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def _duration_minutes(start, end):
    start_time = _parse_time(start)
    end_time = _parse_time(end)
    if start_time is None or end_time is None:
        return ""
    return (end_time - start_time).total_seconds() / 60.0


def _label_set(rows):
    return ",".join(sorted({str(row.get("Label", "")) for row in rows if row.get("Label")}))


def detect_spike_candidates(rows, max_duration_points=2):
    candidates = []
    n = len(rows)
    for start in range(3, max(3, n - 3)):
        baseline_before = _median([_as_float(row.get("FuelLevel")) for row in rows[max(0, start - 4) : start]])
        if baseline_before is None:
            continue

        for end in range(start, min(n - 2, start + max_duration_points - 1) + 1):
            segment = rows[start : end + 1]
            after_rows = rows[end + 1 : min(n, end + 5)]
            baseline_after = _median([_as_float(row.get("FuelLevel")) for row in after_rows])
            if baseline_after is None:
                continue

            fuels = [_as_float(row.get("FuelLevel")) for row in segment]
            if any(value is None for value in fuels):
                continue

            flat = max(_as_float(rows[start].get("flat_jitter_threshold"), 0.8) or 0.8, 0.5)
            spike = max(_as_float(rows[start].get("spike_threshold"), 3.0) or 3.0, flat * 3.0)
            event = max(_as_float(rows[start].get("event_threshold"), 6.0) or 6.0, spike)
            threshold = max(spike * 1.15, event * 0.45, flat * 4.0, 2.0)

            peak = max(fuels)
            trough = min(fuels)
            up_amplitude = peak - baseline_before
            down_amplitude = baseline_before - trough
            if up_amplitude >= down_amplitude:
                direction = "UP"
                extreme_value = peak
                amplitude = up_amplitude
                extreme_offset = fuels.index(peak)
            else:
                direction = "DOWN"
                extreme_value = trough
                amplitude = down_amplitude
                extreme_offset = fuels.index(trough)

            if amplitude < threshold:
                continue

            return_to_before = abs(baseline_after - baseline_before)
            return_gate = max(flat * 2.0, amplitude * 0.25, 1.0)
            if return_to_before > return_gate:
                continue

            # Reject ramps: every point in a spike segment must stay far from the
            # local baseline, and the following baseline must recover.
            if any(abs(fuel - baseline_before) < max(flat * 2.0, amplitude * 0.35) for fuel in fuels):
                continue

            current_labels = {str(row.get("Label", "")) for row in segment}
            if current_labels & {"REFUEL", "DRAIN"}:
                continue

            return_ratio = 1.0 - return_to_before / max(amplitude, 1e-6)
            source_rows = ";".join(str(start + offset) for offset in range(len(segment)))
            candidates.append(
                {
                    "VehicleID": rows[start].get("VehicleID", ""),
                    "SegmentID": rows[start].get("SegmentID", "0"),
                    "Source": rows[start].get("Source", ""),
                    "start_time": rows[start].get("FuelTime", ""),
                    "end_time": rows[end].get("FuelTime", ""),
                    "start_row": start,
                    "end_row": end,
                    "extreme_row": start + extreme_offset,
                    "direction": direction,
                    "baseline_before": baseline_before,
                    "baseline_after": baseline_after,
                    "extreme_value": extreme_value,
                    "amplitude": amplitude,
                    "duration_points": end - start + 1,
                    "duration_minutes": _duration_minutes(rows[start].get("FuelTime"), rows[end].get("FuelTime")),
                    "return_ratio": return_ratio,
                    "suggested_label": "SPIKE",
                    "current_labels": _label_set(segment),
                    "source_rows": source_rows,
                }
            )
            break
    return candidates


def build_candidates(input_path, output_path, max_duration_points=2):
    with open(input_path, encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    groups = {}
    for row_index, row in enumerate(all_rows):
        row["_row_index"] = row_index
        key = (row.get("VehicleID", ""), row.get("SegmentID", "0"))
        groups.setdefault(key, []).append(row)

    candidates = []
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: row.get("FuelTime", ""))
        group_candidates = detect_spike_candidates(group_rows, max_duration_points=max_duration_points)
        for candidate in group_candidates:
            local_source_rows = [int(local_row) for local_row in str(candidate["source_rows"]).split(";") if local_row]
            local_extreme_row = int(candidate["extreme_row"])
            source_rows = []
            for local_row in local_source_rows:
                if local_row:
                    source_rows.append(str(group_rows[local_row]["_row_index"]))
            candidate["start_row"] = int(source_rows[0])
            candidate["end_row"] = int(source_rows[-1])
            candidate["extreme_row"] = int(group_rows[local_extreme_row]["_row_index"]) if source_rows else candidate["extreme_row"]
            candidate["source_rows"] = ";".join(source_rows)
            candidates.append(candidate)

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
        writer.writerows(candidates)

    print(f"Saved spike candidates: {output_path}")
    print(f"Candidate count: {len(candidates):,}")


def main():
    parser = argparse.ArgumentParser(description="Build high-confidence spike event candidates.")
    parser.add_argument("--input", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\all_labeled_points.csv")
    parser.add_argument("--output", default=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset\spike_event_candidates.csv")
    parser.add_argument("--max-duration-points", type=int, default=2)
    args = parser.parse_args()
    build_candidates(args.input, args.output, max_duration_points=args.max_duration_points)


if __name__ == "__main__":
    main()
