import os
import sys
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import glob
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime as new_filter
from scratch.candidate_baseline_filter import filter_ai_enhanced_adaptive_realtime as cand_baseline_filter


def main():
    model_dir = "models/fuel_state_classifier"
    model, metadata = load_fuel_state_classifier(model_dir)
    if model is None or metadata is None:
        raise RuntimeError(f"Cannot load model from {model_dir}")

    files = sorted(glob.glob("TienXuLy/*_processed.csv"))
    vehicle_files = [f for f in files if "Bao_cao_lich_su" not in f]

    print(f"Bắt đầu đánh giá tác động của Nhánh Phục hồi sau nhiễu trên toàn bộ {len(vehicle_files)} xe...")

    exact_match_cars = []
    diff_cars = []

    for idx, fpath in enumerate(vehicle_files, 1):
        v_id = Path(fpath).name.replace("_processed.csv", "")
        df = pd.read_csv(fpath)
        if len(df) == 0 or "FuelLevel" not in df.columns:
            continue

        df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")

        trace_cand = []
        trace_rec = []

        clean_cand = np.array(cand_baseline_filter(df_ai, config={"trace_collector": trace_cand}))
        clean_rec = np.array(new_filter(df_ai, config={"trace_collector": trace_rec}))

        df_cand = pd.DataFrame(trace_cand)
        df_rec = pd.DataFrame(trace_rec)

        diff = clean_rec - clean_cand
        abs_diff = np.abs(diff)
        max_diff = float(np.max(abs_diff))

        rec_activations = int((df_rec["branch_selected"] == "noise_recovery").sum()) if "branch_selected" in df_rec else 0

        if max_diff < 0.05:
            exact_match_cars.append({
                "vehicle_id": v_id,
                "rows": len(df),
                "max_diff": max_diff,
                "rec_activations": rec_activations
            })
        else:
            diff_cars.append({
                "vehicle_id": v_id,
                "rows": len(df),
                "max_diff": max_diff,
                "mean_diff": float(np.mean(abs_diff)),
                "p_gt_1": int(np.sum(abs_diff > 1.0)),
                "rec_activations": rec_activations,
                "max_diff_idx": int(np.argmax(abs_diff)),
                "time_at_max": str(df_ai["FuelTime"].iloc[int(np.argmax(abs_diff))]),
                "raw_at_max": float(df_ai["FuelLevel"].iloc[int(np.argmax(abs_diff))]),
                "clean_cand": float(clean_cand[int(np.argmax(abs_diff))]),
                "clean_rec": float(clean_rec[int(np.argmax(abs_diff))])
            })

    print(f"\n=======================================================")
    print(f"KẾT QUẢ ĐỐI SÁNH TRÊN {len(vehicle_files)} XE:")
    print(f"- Số xe KHỚP TUYỆT ĐỐI (max_diff < 0.05 L): {len(exact_match_cars)} / {len(vehicle_files)}")
    print(f"- Số xe có phân hóa (nhánh phục hồi can thiệp): {len(diff_cars)} / {len(vehicle_files)}")
    print(f"=======================================================\n")

    if exact_match_cars:
        print("Danh sách các xe giữ nguyên vẹn 100% output baseline:")
        for c in exact_match_cars:
            print(f"  Xe {c['vehicle_id']:<15} | Mẫu: {c['rows']:<6} | Max diff: {c['max_diff']:.3f} L")

    if diff_cars:
        print("\nChi tiết các xe có nhánh phục hồi hoạt động:")
        for c in diff_cars:
            print(f"\n* Xe {c['vehicle_id']} (Kích hoạt phục hồi: {c['rec_activations']} nhịp):")
            print(f"  - Max diff: {c['max_diff']:.2f} L | Mean diff: {c['mean_diff']:.3f} L | Số mẫu diff > 1.0L: {c['p_gt_1']}")
            print(f"  - Điểm chênh lớn nhất tại {c['time_at_max']}: Raw = {c['raw_at_max']:.1f} L | Baseline = {c['clean_cand']:.2f} L -> Mới = {c['clean_rec']:.2f} L (Chênh: {c['clean_rec'] - c['clean_cand']:+.2f} L)")


if __name__ == "__main__":
    main()
