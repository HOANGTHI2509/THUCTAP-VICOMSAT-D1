"""Print candidate real-data windows for curating fuel-filter golden tests.

This utility is read-only: it never changes source telemetry files.  The
selected windows are copied manually into tests/fixtures/golden_fuel_segments.json
after visual review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _print_window(name: str, source: Path, frame: pd.DataFrame, start: int, size: int) -> None:
    window = frame.iloc[start : start + size]
    if len(window) < size:
        return
    print(f"\n[{name}] source={source.name} rows={start}:{start + size - 1}")
    print("time=", window["FuelTime"].astype(str).tolist())
    print("raw=", window["FuelLevel"].round(2).tolist())
    print("speed=", window["Speed"].fillna(0.0).round(2).tolist())


def find_candidates(source: Path, size: int) -> None:
    frame = pd.read_csv(source)
    required = {"FuelTime", "FuelLevel", "Speed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")

    fuel = pd.to_numeric(frame["FuelLevel"], errors="coerce")
    speed = pd.to_numeric(frame["Speed"], errors="coerce").fillna(0.0)

    # Short valley: falls materially, then returns close to its earlier level.
    for index in range(3, len(frame) - 6):
        baseline = fuel.iloc[index - 3 : index].median()
        valley = fuel.iloc[index : index + 3].median()
        recovered = fuel.iloc[index + 3 : index + 6].median()
        if baseline - valley >= 8.0 and abs(recovered - baseline) <= 3.0:
            _print_window("u_shape_candidate", source, frame, max(0, index - 3), size)
            break

    # Moving directional consumption: consistently moving and a clear net fall.
    for index in range(0, len(frame) - size):
        raw = fuel.iloc[index : index + size]
        moving = speed.iloc[index : index + size]
        diffs = raw.diff().dropna()
        if (
            (moving > 5.0).mean() >= 0.75
            and raw.iloc[0] - raw.iloc[-1] >= 8.0
            and (diffs <= 1.5).mean() >= 0.75
        ):
            _print_window("moving_consumption_candidate", source, frame, index, size)
            break

    # Strong local noise with little net movement in the window.
    for index in range(0, len(frame) - size):
        raw = fuel.iloc[index : index + size]
        if raw.std() >= 3.0 and abs(raw.iloc[-1] - raw.iloc[0]) <= 3.0:
            _print_window("noise_candidate", source, frame, index, size)
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--size", type=int, default=12)
    args = parser.parse_args()
    for source in args.files:
        find_candidates(source, args.size)


if __name__ == "__main__":
    main()
