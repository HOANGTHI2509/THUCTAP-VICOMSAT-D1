import glob
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.stdout.reconfigure(encoding="utf-8")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "kalman_adaptive_copy",
    os.path.join("src", "core", "filters", "kalman_adaptive copy.py"),
)
kalman_adaptive = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kalman_adaptive
spec.loader.exec_module(kalman_adaptive)


def build_vehicle_profiles(
    input_pattern: str = "TienXuLy/*_processed.csv",
    output_path: str = "artifacts/vehicle_profiles.csv",
) -> None:
    rows = []
    for file_path in sorted(glob.glob(input_pattern)):
        vehicle_id = os.path.basename(file_path).replace("_processed.csv", "")
        df = pd.read_csv(file_path)
        profile = kalman_adaptive.estimate_vehicle_profile(df, vehicle_id)
        rows.append(profile.__dict__)

    if not rows:
        print(f"Khong tim thay file voi pattern: {input_pattern}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Da tao profile cho {len(rows)} xe: {output_path}")


if __name__ == "__main__":
    build_vehicle_profiles()
