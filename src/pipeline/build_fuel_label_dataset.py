import csv
import math
import os
import re
import statistics
import sys
import zipfile
from collections import Counter, deque
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET


sys.stdout.reconfigure(encoding="utf-8")

MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

TIENXULY_TRAIN_FILES = [
    "12H-04470_processed.csv",
    "19H-06956_processed.csv",
    "20B-27762_processed.csv",
    "21C-04064_processed.csv",
    "21H-02058_processed.csv",
    "21H-03052_processed.csv",
    "21H-03221_processed.csv",
    "24H-03439_processed.csv",
    "24H-04058_processed.csv",
    "24H-05088_processed.csv",
    "29E-44284_processed.csv",
    "29E-45520_processed.csv",
    "29E-45560_processed.csv",
    "29E-51878_processed.csv",
    "29H-41394_processed.csv",
    "35H-09245_processed.csv",
    "35H-14767_processed.csv",
    "36C-31893_processed.csv",
    "92H-02653_processed.csv",
    "92H-07095_processed.csv",
]

# Held-out vehicle: never included in train. It provides a license-plate
# evaluation case in addition to the GPS test vehicle (Car 5).
TIENXULY_TEST_FILES = [
    "90H-03494_processed.csv",
    "92H-02687_processed.csv",
]


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        number = float(str(value).replace(",", "."))
        if math.isnan(number):
            return None
        return number
    except Exception:
        return None


def _excel_datetime(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
        if number > 20000:
            return datetime(1899, 12, 30) + timedelta(days=number)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _percentile(values, percentile):
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        return None
    k = (len(values) - 1) * percentile / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def _column_index(cell_ref):
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    number = 0
    for ch in letters:
        number = number * 26 + ord(ch) - 64
    return number - 1


def _haversine_m(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    radius = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_xlsx_rows(path, skip_sheets=None):
    skip_sheets = set(skip_sheets or [])
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter(MAIN_NS + "si"):
                shared.append("".join((t.text or "") for t in item.iter(MAIN_NS + "t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheets = [
            (sheet.attrib["name"], sheet.attrib[REL_NS + "id"])
            for sheet in workbook.find(MAIN_NS + "sheets")
        ]

        def cell_value(cell):
            value = cell.find(MAIN_NS + "v")
            if value is None:
                return ""
            text = value.text or ""
            if cell.attrib.get("t") == "s" and text.isdigit():
                index = int(text)
                return shared[index] if index < len(shared) else text
            return text

        for sheet_name, rel_id in sheets:
            if sheet_name in skip_sheets:
                continue
            sheet_path = "xl/" + rel_targets[rel_id]
            headers = {}
            rows = []
            for _, row in ET.iterparse(archive.open(sheet_path), events=("end",)):
                if row.tag != MAIN_NS + "row":
                    continue
                row_number = int(row.attrib.get("r", "0"))
                cells = {_column_index(cell.attrib["r"]): cell_value(cell) for cell in row.iter(MAIN_NS + "c")}
                if row_number == 1:
                    headers = {str(value): index for index, value in cells.items()}
                else:
                    fuel_time = _excel_datetime(cells.get(headers.get("FuelTime")))
                    fuel_level = _to_float(cells.get(headers.get("FuelLevel")))
                    lat = _to_float(cells.get(headers.get("Lat")))
                    lng = _to_float(cells.get(headers.get("Lng")))
                    speed = _to_float(cells.get(headers.get("Speed")))
                    if fuel_time is not None and fuel_level is not None:
                        rows.append(
                            {
                                "VehicleID": sheet_name,
                                "FuelTime": fuel_time,
                                "FuelLevel": fuel_level,
                                "Lat": lat,
                                "Lng": lng,
                                "Speed": speed if speed is not None else 0.0,
                            }
                        )
                row.clear()
            rows.sort(key=lambda item: item["FuelTime"])
            yield sheet_name, rows


def _parse_datetime(value):
    if value is None or value == "":
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(str(value)[:19], fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _load_tienxuly_rows(folder, filenames):
    for filename in filenames:
        path = os.path.join(folder, filename)
        rows = []
        if not os.path.exists(path):
            print(f"Skip missing TienXuLy file: {path}")
            continue
        vehicle_id = filename.replace("_processed.csv", "")
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                fuel_time = _parse_datetime(raw.get("FuelTime"))
                fuel_level = _to_float(raw.get("FuelLevel"))
                speed = _to_float(raw.get("Speed"))
                if fuel_time is None or fuel_level is None:
                    continue
                rows.append(
                    {
                        "VehicleID": vehicle_id,
                        "FuelTime": fuel_time,
                        "FuelLevel": fuel_level,
                        "Lat": None,
                        "Lng": None,
                        "Speed": speed if speed is not None else 0.0,
                    }
                )
        rows.sort(key=lambda item: item["FuelTime"])
        yield vehicle_id, rows


def _estimate_profile(rows):
    fuels = [row["FuelLevel"] for row in rows if row["FuelLevel"] > 0]
    capacity = _percentile(fuels, 99.5) or max(fuels or [200.0])
    deltas = [abs(rows[i]["FuelLevel"] - rows[i - 1]["FuelLevel"]) for i in range(1, len(rows))]
    small_cut = _percentile(deltas, 60) or 0.0
    small_deltas = [value for value in deltas if value <= small_cut]
    noise = 1.4826 * statistics.median(small_deltas) if small_deltas else max(0.5, 0.002 * capacity)
    if noise <= 0 or math.isnan(noise):
        noise = max(0.5, 0.002 * capacity)
    return {
        "capacity_est": capacity,
        "noise_sigma_liters": noise,
        "flat_jitter_threshold": max(0.5, 0.003 * capacity, 2.0 * noise),
        "spike_threshold": max(2.0, 0.012 * capacity, 4.0 * noise),
        "event_threshold": max(5.0, 0.035 * capacity, 8.0 * noise),
    }


def _speed_band(speed):
    if speed <= 3:
        return "STOPPED"
    if speed <= 10:
        return "CRAWLING"
    if speed <= 30:
        return "LOW_SPEED"
    if speed <= 60:
        return "NORMAL_SPEED"
    return "HIGH_SPEED"


def _add_features_and_labels(rows):
    profile = _estimate_profile(rows)
    rolling_values = deque(maxlen=12)
    prev = None
    enriched = []

    for index, row in enumerate(rows):
        fuel = row["FuelLevel"]
        speed = row["Speed"]
        if prev is None:
            gap = 0.0
            delta = 0.0
            distance = 0.0
            gps_speed = 0.0
        else:
            gap = (row["FuelTime"] - prev["FuelTime"]).total_seconds() / 60.0
            if gap <= 0 or gap > 1440:
                gap = 0.0
            delta = fuel - prev["FuelLevel"]
            distance = _haversine_m(prev["Lat"], prev["Lng"], row["Lat"], row["Lng"])
            if distance is None:
                distance = 0.0
            gps_speed = (distance / 1000.0) / (gap / 60.0) if gap > 0 else 0.0

        rolling_values.append(fuel)
        rolling_std = statistics.pstdev(rolling_values) if len(rolling_values) >= 2 else 0.0
        motion_speed = max(speed, gps_speed)
        enriched.append(
            {
                **row,
                "TimeGapMinutes": gap,
                "DeltaFuel": delta,
                "AbsDeltaFuel": abs(delta),
                "RollingStd12": rolling_std,
                "DistanceMeters": distance,
                "GpsSpeedKmh": gps_speed,
                "MotionSpeedKmh": motion_speed,
                "HasGPS": int(row["Lat"] is not None and row["Lng"] is not None),
                "SpeedBand": _speed_band(motion_speed),
                "Label": "UNKNOWN",
                **profile,
            }
        )
        prev = row

    flat = profile["flat_jitter_threshold"]
    spike = profile["spike_threshold"]
    event = profile["event_threshold"]
    capacity = profile["capacity_est"]
    low_fuel_cut = max(5.0, 0.03 * capacity)

    for index, row in enumerate(enriched):
        fuel = row["FuelLevel"]
        delta = row["DeltaFuel"]
        speed = row["MotionSpeedKmh"]
        rolling_std = row["RollingStd12"]
        prev_fuel = enriched[index - 1]["FuelLevel"] if index > 0 else fuel
        stable_band = max(flat * 1.75, 2.0 * profile["noise_sigma_liters"], 0.8)
        sloshing_band = max(spike, flat * 2.2, 4.0 * profile["noise_sigma_liters"])
        prev_delta = enriched[index - 1]["DeltaFuel"] if index > 0 else 0.0
        prev2_fuel = enriched[index - 2]["FuelLevel"] if index > 1 else prev_fuel
        prev_fuels = [item["FuelLevel"] for item in enriched[max(0, index - 3) : index]]
        next_fuels = [item["FuelLevel"] for item in enriched[index + 1 : index + 4]]
        next_fuels_long = [item["FuelLevel"] for item in enriched[index + 1 : index + 6]]
        local_fuels = prev_fuels + [fuel] + next_fuels
        local_deltas = [enriched[i]["DeltaFuel"] for i in range(max(1, index - 3), min(len(enriched), index + 4))]
        future_median = statistics.median(next_fuels) if next_fuels else fuel
        future_median_long = statistics.median(next_fuels_long) if next_fuels_long else future_median
        prev_median3 = statistics.median(prev_fuels) if prev_fuels else prev_fuel
        future_median3 = future_median
        future_median5 = future_median_long
        returns_to_previous = abs(future_median - prev_fuel) <= max(flat, 0.25 * abs(delta))
        holds_new_level = abs(future_median_long - fuel) <= max(flat, 0.25 * abs(delta))
        local_range = max(local_fuels) - min(local_fuels) if len(local_fuels) >= 3 else 0.0
        signed_moves = [1 if value > flat else -1 if value < -flat else 0 for value in local_deltas]
        signed_moves = [value for value in signed_moves if value != 0]
        sign_changes = sum(1 for left, right in zip(signed_moves, signed_moves[1:]) if left != right)
        volatile_cluster = (
            rolling_std > sloshing_band
            or (local_range >= event and sign_changes >= 2)
            or (local_range >= max(spike * 1.4, flat * 4.0) and rolling_std > flat * 2.0)
        )
        recovery_after_drop = (
            prev_delta <= -spike
            and delta >= spike
            and abs(fuel - prev2_fuel) <= max(flat * 2.0, 0.35 * abs(prev_delta))
        )
        isolated_drop = delta <= -spike and returns_to_previous
        isolated_jump = delta >= spike and returns_to_previous
        transient_window = enriched[max(0, index - 2) : min(len(enriched), index + 6)]
        transient_fuels = [item["FuelLevel"] for item in transient_window]
        transient_range = max(transient_fuels) - min(transient_fuels) if len(transient_fuels) >= 5 else 0.0
        pre_level = prev_median3
        post_level = future_median_long
        return_to_prev_level = abs(post_level - pre_level)
        returns_to_local_baseline = return_to_prev_level <= max(flat * 2.5, transient_range * 0.30)
        local_range5 = transient_range
        local_range7_fuels = [item["FuelLevel"] for item in enriched[max(0, index - 3) : min(len(enriched), index + 4)]]
        local_range7 = max(local_range7_fuels) - min(local_range7_fuels) if len(local_range7_fuels) >= 5 else local_range5
        peak_reversal_flag = int(delta > flat * 1.5 and next_fuels and (next_fuels[0] - fuel) < -flat * 1.5)
        valley_reversal_flag = int(delta < -flat * 1.5 and next_fuels and (next_fuels[0] - fuel) > flat * 1.5)
        transient_score = 0.0
        if local_range5 > 0:
            transient_score = max(0.0, 1.0 - return_to_prev_level / max(local_range5, 1e-6))
        short_bump_cluster = (
            transient_range >= max(event * 0.60, spike * 1.20, flat * 4.0)
            and returns_to_local_baseline
            and abs(fuel - pre_level) >= max(spike * 0.80, flat * 3.0)
            and abs(delta) < max(event * 1.25, transient_range)
        )
        transient_direction = 1 if fuel > pre_level else -1 if fuel < pre_level else 0

        if fuel <= 0:
            label = "INVALID"
        elif prev_fuel <= low_fuel_cut and delta >= event:
            label = "UNKNOWN"
        elif fuel <= low_fuel_cut and prev_fuel > max(30.0, 0.20 * capacity):
            label = "DROPOUT"
        elif short_bump_cluster and transient_direction > 0:
            label = "TRANSIENT_UP_NOISE"
        elif short_bump_cluster and transient_direction < 0:
            label = "TRANSIENT_DOWN_NOISE"
        elif recovery_after_drop or isolated_drop or isolated_jump:
            label = "SPIKE"
        elif delta >= event and holds_new_level and not recovery_after_drop:
            label = "REFUEL"
        elif delta <= -event and holds_new_level and fuel > low_fuel_cut:
            label = "DRAIN"
        elif volatile_cluster and abs(delta) < event:
            label = "SLOSHING_NOISE"
        elif delta < -0.5 * flat and speed > 3 and abs(delta) < event:
            label = "CONSUMPTION"
        elif abs(delta) <= stable_band and rolling_std <= max(stable_band * 2.0, 3.0 * profile["noise_sigma_liters"]):
            label = "STABLE_JITTER"
        else:
            label = "UNKNOWN"

        row["Label"] = label
        row["FuelPct"] = fuel / capacity if capacity else 0.0
        row["DeltaPct"] = delta / capacity if capacity else 0.0
        row["RollingStdPct"] = rolling_std / capacity if capacity else 0.0
        row["DeltaOverNoise"] = delta / profile["noise_sigma_liters"] if profile["noise_sigma_liters"] else 0.0
        row["PrevMedian3"] = prev_median3
        row["FutureMedian3"] = future_median3
        row["FutureMedian5"] = future_median5
        row["ReturnToPrevLevel"] = return_to_prev_level
        row["LocalRange5"] = local_range5
        row["LocalRange7"] = local_range7
        row["PeakReversalFlag"] = peak_reversal_flag
        row["ValleyReversalFlag"] = valley_reversal_flag
        row["TransientScore"] = transient_score

    return enriched


def _split_name(vehicle_id, order_index, total_for_vehicle):
    if vehicle_id in {"Car 1", "Car 2"}:
        return "train"
    if vehicle_id == "Car 3":
        return "train" if order_index < int(total_for_vehicle * 0.7) else "val"
    if vehicle_id == "Car 5":
        return "test"
    return "unused"


def build_dataset(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "Split",
        "Source",
        "VehicleID",
        "FuelTime",
        "FuelLevel",
        "FuelPct",
        "Lat",
        "Lng",
        "Speed",
        "MotionSpeedKmh",
        "SpeedBand",
        "TimeGapMinutes",
        "DeltaFuel",
        "DeltaPct",
        "AbsDeltaFuel",
        "DeltaOverNoise",
        "RollingStd12",
        "RollingStdPct",
        "DistanceMeters",
        "GpsSpeedKmh",
        "HasGPS",
        "capacity_est",
        "noise_sigma_liters",
        "flat_jitter_threshold",
        "spike_threshold",
        "event_threshold",
        "PrevMedian3",
        "FutureMedian3",
        "FutureMedian5",
        "ReturnToPrevLevel",
        "LocalRange5",
        "LocalRange7",
        "PeakReversalFlag",
        "ValleyReversalFlag",
        "TransientScore",
        "Label",
    ]
    writers = {}
    files = {}
    for name in ["all_labeled_points", "train", "val", "test"]:
        handle = open(os.path.join(output_dir, f"{name}.csv"), "w", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        files[name] = handle
        writers[name] = writer

    summary_rows = []
    try:
        for vehicle_id, rows in _load_xlsx_rows(input_path, skip_sheets={"Car 4"}):
            enriched = _add_features_and_labels(rows)
            counts = Counter()
            split_counts = Counter()
            total = len(enriched)
            for index, row in enumerate(enriched):
                split = _split_name(vehicle_id, index, total)
                row_out = {key: row.get(key, "") for key in fieldnames}
                row_out["Split"] = split
                row_out["Source"] = "CarFuelHistory.xlsx"
                row_out["FuelTime"] = row["FuelTime"].isoformat(sep=" ")
                writers["all_labeled_points"].writerow(row_out)
                if split in {"train", "val", "test"}:
                    writers[split].writerow(row_out)
                counts[row["Label"]] += 1
                split_counts[split] += 1

            summary_rows.append(
                {
                    "VehicleID": vehicle_id,
                    "Rows": total,
                    "SplitCounts": dict(split_counts),
                    "LabelCounts": dict(counts),
                }
            )
            print(f"{vehicle_id}: rows={total:,} labels={dict(counts)} splits={dict(split_counts)}")

        tienxuly_folder = r"D:\THUCTAP_VICOMSAT\TienXuLy"
        for vehicle_id, rows in _load_tienxuly_rows(tienxuly_folder, TIENXULY_TRAIN_FILES):
            enriched = _add_features_and_labels(rows)
            counts = Counter()
            split_counts = Counter()
            total = len(enriched)
            for row in enriched:
                row_out = {key: row.get(key, "") for key in fieldnames}
                row_out["Split"] = "train"
                row_out["Source"] = "TienXuLy"
                row_out["FuelTime"] = row["FuelTime"].isoformat(sep=" ")
                writers["all_labeled_points"].writerow(row_out)
                writers["train"].writerow(row_out)
                counts[row["Label"]] += 1
                split_counts["train"] += 1

            summary_rows.append(
                {
                    "VehicleID": vehicle_id,
                    "Rows": total,
                    "SplitCounts": dict(split_counts),
                    "LabelCounts": dict(counts),
                }
            )
            print(f"{vehicle_id}: rows={total:,} labels={dict(counts)} splits={dict(split_counts)}")

        for vehicle_id, rows in _load_tienxuly_rows(tienxuly_folder, TIENXULY_TEST_FILES):
            enriched = _add_features_and_labels(rows)
            counts = Counter()
            split_counts = Counter()
            total = len(enriched)
            for row in enriched:
                row_out = {key: row.get(key, "") for key in fieldnames}
                row_out["Split"] = "test"
                row_out["Source"] = "TienXuLy"
                row_out["FuelTime"] = row["FuelTime"].isoformat(sep=" ")
                writers["all_labeled_points"].writerow(row_out)
                writers["test"].writerow(row_out)
                counts[row["Label"]] += 1
                split_counts["test"] += 1

            summary_rows.append(
                {
                    "VehicleID": vehicle_id,
                    "Rows": total,
                    "SplitCounts": dict(split_counts),
                    "LabelCounts": dict(counts),
                }
            )
            print(f"{vehicle_id}: rows={total:,} labels={dict(counts)} splits={dict(split_counts)}")
    finally:
        for handle in files.values():
            handle.close()

    with open(os.path.join(output_dir, "label_summary.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["VehicleID", "Rows", "SplitCounts", "LabelCounts"])
        for row in summary_rows:
            writer.writerow([row["VehicleID"], row["Rows"], row["SplitCounts"], row["LabelCounts"]])


if __name__ == "__main__":
    build_dataset(
        input_path=r"D:\THUCTAP_VICOMSAT\CarFuelHistory.xlsx",
        output_dir=r"D:\THUCTAP_VICOMSAT\data\fuel_label_dataset",
    )
