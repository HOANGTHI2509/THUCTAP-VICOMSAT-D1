"""Create terrain-aware fuel charts for the two TestDoDoc traces.

The coloured bands are *warnings*, not a terrain correction: a blue band marks
an uphill GPS leg and a green band marks a downhill leg where the DEM-derived
road grade reaches +/-3 percent.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results"
CASES = ["TEST DO DOC", "2026-08-27T00-48_export"]
WIDTH, HEIGHT = 1600, 980
LEFT, RIGHT = 92, 70
PANEL_H = 235
TOPS = (88, 406, 724)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def points(values: pd.Series, top: float, ymin: float, ymax: float) -> str:
    x0, x1 = LEFT, WIDTH - RIGHT
    xs = np.linspace(x0, x1, len(values))
    safe = values.clip(ymin, ymax).fillna(ymin)
    ys = top + PANEL_H - (safe - ymin) / max(ymax - ymin, 1e-9) * PANEL_H
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def y_of(value: float, top: float, ymin: float, ymax: float) -> float:
    return top + PANEL_H - (value - ymin) / max(ymax - ymin, 1e-9) * PANEL_H


def slope_bands(df: pd.DataFrame) -> list[tuple[int, int, str]]:
    # Join immediately adjacent suspect legs into one leg-sized coloured band.
    classes = np.where(df["RoadGradePct"] >= 3, "uphill", np.where(df["RoadGradePct"] <= -3, "downhill", ""))
    bands: list[tuple[int, int, str]] = []
    start, label = None, ""
    for i, value in enumerate(classes):
        if value and value == label:
            continue
        if start is not None:
            bands.append((start, i - 1, label))
            start = None
        if value:
            start, label = i, value
        else:
            label = ""
    if start is not None:
        bands.append((start, len(classes) - 1, label))
    return bands


def make_chart(case: str) -> Path:
    terrain = pd.read_csv(RESULTS / f"{case}_terrain.csv")
    clean = pd.read_csv(RESULTS / f"{case}_realtime_clean.csv")
    terrain_aware_path = RESULTS / f"{case}_terrain_aware_clean.csv"
    terrain_aware = pd.read_csv(terrain_aware_path) if terrain_aware_path.exists() else None
    for col in ("FuelLevel", "ElevationMSmoothed", "RoadGradePct"):
        terrain[col] = numeric(terrain[col])
    clean_fuel = numeric(clean["CleanFuelRealtime"])
    terrain_aware_fuel = numeric(terrain_aware["CleanFuelTerrainAware"]) if terrain_aware is not None else None
    if len(clean_fuel) != len(terrain):
        clean_fuel = clean_fuel.reindex(range(len(terrain)))
    n = len(terrain)
    plot_width = WIDTH - LEFT - RIGHT

    fuel_values = np.r_[terrain["FuelLevel"], clean_fuel]
    if terrain_aware_fuel is not None:
        fuel_values = np.r_[fuel_values, terrain_aware_fuel]
    fuel_min = float(np.nanmin(fuel_values))
    fuel_max = float(np.nanmax(fuel_values))
    fuel_pad = max((fuel_max - fuel_min) * .08, 1.0)
    fuel_min, fuel_max = fuel_min - fuel_pad, fuel_max + fuel_pad
    elev_min, elev_max = terrain["ElevationMSmoothed"].min(), terrain["ElevationMSmoothed"].max()
    elev_pad = max((elev_max - elev_min) * .10, 3.0)
    elev_min, elev_max = elev_min - elev_pad, elev_max + elev_pad
    grade_lim = max(5.0, float(terrain["RoadGradePct"].abs().quantile(.99)))

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#233047}.title{font-size:27px;font-weight:700}.sub{font-size:14px;fill:#68758b}.axis{font-size:12px;fill:#68758b}.panel{font-size:17px;font-weight:700}</style>',
        f'<text x="{LEFT}" y="38" class="title">Fuel vs terrain — {html.escape(case)}</text>',
        '<text x="92" y="61" class="sub">Blue = uphill suspect (grade ≥ 3%); green = downhill suspect (grade ≤ −3%). Shading is a warning feature, not a fuel correction.</text>',
    ]
    for top, label, ymin, ymax, unit in [
        (TOPS[0], "Fuel level", fuel_min, fuel_max, "L"),
        (TOPS[1], "DEM elevation (smoothed)", elev_min, elev_max, "m"),
        (TOPS[2], "Road grade", -grade_lim, grade_lim, "%"),
    ]:
        svg.append(f'<text x="{LEFT}" y="{top-13}" class="panel">{label}</text>')
        for fraction in (0, .25, .5, .75, 1):
            y = top + PANEL_H * fraction
            val = ymax - (ymax-ymin)*fraction
            svg.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH-RIGHT}" y2="{y:.1f}" stroke="#dce5f0"/>')
            svg.append(f'<text x="{LEFT-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{val:.1f}</text>')
        svg.append(f'<text x="26" y="{top+PANEL_H/2:.1f}" class="axis">{unit}</text>')

    # Terrain bands span all panels, making it easy to compare fuel behaviour.
    for first, last, label in slope_bands(terrain):
        x = LEFT + max(0, first - .5) / max(n - 1, 1) * plot_width
        x2 = LEFT + min(n - 1, last + .5) / max(n - 1, 1) * plot_width
        colour = "#60a5fa" if label == "uphill" else "#34d399"
        svg.append(f'<rect x="{x:.1f}" y="{TOPS[0]-2}" width="{max(x2-x, 2):.1f}" height="{TOPS[2]+PANEL_H-TOPS[0]+4}" fill="{colour}" opacity=".32"/>')

    svg.extend([
        f'<polyline points="{points(terrain["FuelLevel"], TOPS[0], fuel_min, fuel_max)}" fill="none" stroke="#ef4444" stroke-width="1.4"/>',
        f'<polyline points="{points(clean_fuel, TOPS[0], fuel_min, fuel_max)}" fill="none" stroke="#f97316" stroke-width="2.4"/>',
        f'<polyline points="{points(terrain["ElevationMSmoothed"], TOPS[1], elev_min, elev_max)}" fill="none" stroke="#7c3aed" stroke-width="2"/>',
        f'<line x1="{LEFT}" y1="{y_of(0, TOPS[2], -grade_lim, grade_lim):.1f}" x2="{WIDTH-RIGHT}" y2="{y_of(0, TOPS[2], -grade_lim, grade_lim):.1f}" stroke="#94a3b8" stroke-width="1"/>',
        f'<polyline points="{points(terrain["RoadGradePct"].fillna(0), TOPS[2], -grade_lim, grade_lim)}" fill="none" stroke="#475569" stroke-width="1.5"/>',
        f'<line x1="{LEFT}" y1="{y_of(3, TOPS[2], -grade_lim, grade_lim):.1f}" x2="{WIDTH-RIGHT}" y2="{y_of(3, TOPS[2], -grade_lim, grade_lim):.1f}" stroke="#60a5fa" stroke-dasharray="5 4"/>',
        f'<line x1="{LEFT}" y1="{y_of(-3, TOPS[2], -grade_lim, grade_lim):.1f}" x2="{WIDTH-RIGHT}" y2="{y_of(-3, TOPS[2], -grade_lim, grade_lim):.1f}" stroke="#34d399" stroke-dasharray="5 4"/>',
        '<line x1="1180" y1="31" x2="1220" y2="31" stroke="#ef4444" stroke-width="2"/><text x="1228" y="36" class="axis">Raw fuel</text>',
        '<line x1="1305" y1="31" x2="1345" y2="31" stroke="#f97316" stroke-width="3"/><text x="1353" y="36" class="axis">Clean realtime</text>',
        f'<text x="{LEFT}" y="{HEIGHT-16}" class="axis">Samples: {n:,} · elevation source: Open-Meteo DEM 90 m · GPS coordinates were auto-swapped in this export</text>',
        '</svg>',
    ])
    if terrain_aware_fuel is not None:
        svg.insert(-1, f'<polyline points="{points(terrain_aware_fuel, TOPS[0], fuel_min, fuel_max)}" fill="none" stroke="#0f766e" stroke-width="2.1"/>')
        svg.insert(-1, '<line x1="1430" y1="31" x2="1470" y2="31" stroke="#0f766e" stroke-width="3"/><text x="1478" y="36" class="axis">Terrain-aware</text>')
    output = RESULTS / f"{case}_terrain_overlay.svg"
    output.write_text("\n".join(svg), encoding="utf-8")
    return output


if __name__ == "__main__":
    for test_case in CASES:
        print(f"Saved {make_chart(test_case)}")
