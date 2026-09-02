"""Add DEM elevation and road-grade features to the two TestDoDoc traces.

This test-time adapter uses Open-Meteo's 90 m DEM endpoint because no local
SRTM .hgt tiles are present in the workspace.  The output schema is source
neutral, so production can replace ``lookup_elevations`` with local SRTM.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
INPUT_FILES = [BASE_DIR / "TEST DO DOC.csv", BASE_DIR / "2026-08-27T00-48_export.csv"]
CACHE_PATH = OUTPUT_DIR / "elevation_cache_open_meteo_90m.json"
API_URL = "https://api.open-meteo.com/v1/elevation"
BATCH_SIZE = 100


def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    for column in ["Lat", "Lng", "Speed"]:
        if column not in df.columns:
            raise ValueError(f"{path.name} missing {column}")
        df[column] = pd.to_numeric(df[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
    df = df.dropna(subset=["FuelTime", "Lat", "Lng"]).sort_values("FuelTime", kind="stable").reset_index(drop=True)

    # VICOMSAT exports in these two files have longitude in Lat and latitude
    # in Lng. Correct only rows that are physically impossible as latitude.
    swapped = df["Lat"].abs().gt(90) & df["Lng"].abs().le(90)
    df["GpsCoordinateSwapped"] = swapped.astype(int)
    df.loc[swapped, ["Lat", "Lng"]] = df.loc[swapped, ["Lng", "Lat"]].to_numpy()
    valid = df["Lat"].between(-90, 90) & df["Lng"].between(-180, 180)
    df.loc[~valid, ["Lat", "Lng"]] = np.nan
    return df


def coordinate_key(lat: float, lng: float) -> str:
    # 4 decimals is roughly an 11 m grid: enough for a 90 m DEM and greatly
    # reduces remote calls for dense GPS tracks.
    return f"{lat:.4f},{lng:.4f}"


def lookup_elevations(keys: list[str], cache: dict[str, float]) -> None:
    missing = [key for key in keys if key not in cache]
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        latitudes = ",".join(key.split(",")[0] for key in batch)
        longitudes = ",".join(key.split(",")[1] for key in batch)
        url = f"{API_URL}?{urlencode({'latitude': latitudes, 'longitude': longitudes})}"
        for attempt in range(4):
            try:
                with urlopen(url, timeout=30) as response:  # nosec B310 - fixed HTTPS endpoint
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as error:
                if error.code != 429 or attempt == 3:
                    raise
                wait_seconds = 5 * (attempt + 1)
                print(f"Elevation API rate-limited; retrying in {wait_seconds}s")
                time.sleep(wait_seconds)
        elevations = payload.get("elevation", [])
        if len(elevations) != len(batch):
            raise RuntimeError(f"Elevation response mismatch: requested {len(batch)}, received {len(elevations)}")
        cache.update({key: float(value) for key, value in zip(batch, elevations) if value is not None})
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"Fetched elevation {min(start + BATCH_SIZE, len(missing))}/{len(missing)}")
        time.sleep(1.0)


def haversine_m(lat1: pd.Series, lng1: pd.Series, lat2: pd.Series, lng2: pd.Series) -> pd.Series:
    radius_m = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lng2 - lng1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_terrain_features(df: pd.DataFrame, cache: dict[str, float]) -> pd.DataFrame:
    result = df.copy()
    keys = [coordinate_key(lat, lng) if pd.notna(lat) and pd.notna(lng) else None for lat, lng in zip(result["Lat"], result["Lng"])]
    result["ElevationM"] = [cache.get(key, np.nan) if key else np.nan for key in keys]
    # A local median limits 90 m DEM quantisation before calculating grade.
    result["ElevationMSmoothed"] = result["ElevationM"].rolling(5, center=True, min_periods=1).median()
    result["DistanceMetersForSlope"] = haversine_m(
        result["Lat"].shift(), result["Lng"].shift(), result["Lat"], result["Lng"]
    )
    result["DeltaElevationM"] = result["ElevationMSmoothed"].diff()
    usable_distance = result["DistanceMetersForSlope"].where(result["DistanceMetersForSlope"] >= 25.0)
    result["RoadGradePct"] = (100.0 * result["DeltaElevationM"] / usable_distance).clip(-25.0, 25.0)
    result["SlopeDirection"] = np.select(
        [result["RoadGradePct"] >= 3.0, result["RoadGradePct"] <= -3.0],
        ["UPHILL", "DOWNHILL"],
        default="FLAT_OR_UNKNOWN",
    )
    result["SlopeSuspect"] = (result["RoadGradePct"].abs() >= 3.0).astype(int)
    result["ElevationSource"] = "open_meteo_dem_90m"
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    traces = [(path, load_trace(path)) for path in INPUT_FILES]
    all_keys = sorted({coordinate_key(lat, lng) for _, df in traces for lat, lng in zip(df["Lat"], df["Lng"]) if pd.notna(lat) and pd.notna(lng)})
    lookup_elevations(all_keys, cache)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    report = ["# Terrain enrichment report", "", "Elevation source: Open-Meteo 90 m DEM", ""]
    for path, df in traces:
        enriched = add_terrain_features(df, cache)
        output_path = OUTPUT_DIR / f"{path.stem}_terrain.csv"
        enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
        report.extend([
            f"## {path.name}", "",
            f"- Rows: {len(enriched):,}",
            f"- Coordinates auto-swapped: {int(enriched['GpsCoordinateSwapped'].sum()):,}",
            f"- Elevation range: {enriched['ElevationM'].min():.1f}–{enriched['ElevationM'].max():.1f} m",
            f"- SlopeSuspect rows (|grade| ≥ 3%): {int(enriched['SlopeSuspect'].sum()):,}",
            f"- Output: `{output_path.name}`", "",
        ])
        print(f"Saved {output_path}")
    (OUTPUT_DIR / "terrain_enrichment_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
