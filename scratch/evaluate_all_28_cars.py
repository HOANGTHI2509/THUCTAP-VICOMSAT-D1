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
from scratch.legacy_adaptive_filter import filter_ai_enhanced_adaptive_realtime as legacy_filter


def main():
    model_dir = "models/fuel_state_classifier"
    model, metadata = load_fuel_state_classifier(model_dir)
    if model is None or metadata is None:
        raise RuntimeError(f"Cannot load model from {model_dir}")

    files = sorted(glob.glob("TienXuLy/*_processed.csv"))
    # Loại bỏ file báo cáo tổng hợp
    vehicle_files = [f for f in files if "Bao_cao_lich_su" not in f]

    print(f"Bắt đầu đánh giá trên {len(vehicle_files)} xe...")

    summary_rows = []
    top_diff_segments = []
    u_shape_segments = []

    for idx, fpath in enumerate(vehicle_files, 1):
        v_id = Path(fpath).name.replace("_processed.csv", "")
        t0 = time.time()
        df = pd.read_csv(fpath)
        if len(df) == 0 or "FuelLevel" not in df.columns:
            continue

        # Causal feature enrichment
        df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")

        # Thu thập trace cho cả 2 bản
        trace_old = []
        trace_new = []
        cfg_old = {"source_col": "FuelLevel", "trace_collector": trace_old}
        cfg_new = {"source_col": "FuelLevel", "trace_collector": trace_new}

        clean_old = legacy_filter(df_ai, config=cfg_old)
        clean_new = new_filter(df_ai, config=cfg_new)

        t_elapsed = time.time() - t0

        df_old = pd.DataFrame(trace_old)
        df_new = pd.DataFrame(trace_new)

        arr_old = np.array(clean_old)
        arr_new = np.array(clean_new)
        diff = arr_new - arr_old
        abs_diff = np.abs(diff)

        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        p_gt_1 = int(np.sum(abs_diff > 1.0))
        p_gt_5 = int(np.sum(abs_diff > 5.0))

        # Thống kê xác nhận tăng
        old_refuels = int((df_old["branch_selected"] == "is_refuel").sum()) if "branch_selected" in df_old else 0
        new_refuels = int((df_new["branch_selected"] == "is_refuel").sum()) if "branch_selected" in df_new else 0

        # Thống kê ứng viên mới
        # Trong trace mới:
        # candidate_open: khi candidate_count == 1
        # candidate_cancel: khi branch_selected == "candidate_canceled" hoặc "candidate_step_down_reset"
        cand_opens = 0
        cand_resets = 0
        max_cand_dur = 0.0
        if "candidate_count" in df_new:
            # Đếm số lần candidate_count chuyển từ 0 sang 1
            c_counts = df_new["candidate_count"].to_numpy()
            cand_opens = int(np.sum((c_counts[1:] >= 1) & (c_counts[:-1] == 0)))
            if c_counts[0] >= 1:
                cand_opens += 1
            if "candidate_duration_min" in df_new:
                max_cand_dur = float(df_new["candidate_duration_min"].max())

        # Điểm lệch lớn nhất
        max_idx = int(np.argmax(abs_diff))
        max_time = str(df_ai["FuelTime"].iloc[max_idx]) if max_idx < len(df_ai) else ""
        max_raw = float(df_ai["FuelLevel"].iloc[max_idx]) if max_idx < len(df_ai) else 0.0
        max_o = float(arr_old[max_idx])
        max_n = float(arr_new[max_idx])

        summary_rows.append({
            "vehicle_id": v_id,
            "rows": len(df),
            "max_diff": round(max_diff, 2),
            "mean_diff": round(mean_diff, 3),
            "points_diff_gt_1L": p_gt_1,
            "points_diff_gt_5L": p_gt_5,
            "old_refuel_count": old_refuels,
            "new_refuel_count": new_refuels,
            "cand_opens": cand_opens,
            "max_cand_duration_min": round(max_cand_dur, 1),
            "max_diff_time": max_time,
            "max_diff_raw": round(max_raw, 1),
            "max_diff_old": round(max_o, 2),
            "max_diff_new": round(max_n, 2),
            "time_sec": round(t_elapsed, 2),
        })

        print(f"[{idx:2d}/{len(vehicle_files)}] {v_id:10s} | Rows: {len(df):4d} | MaxDiff: {max_diff:6.2f} L | Refuels (Old/New): {old_refuels:2d}/{new_refuels:2d} | CandOpens: {cand_opens:2d} | Time: {t_elapsed:.1f}s")

        # Nếu có lệch lớn (> 10 L), ghi lại đoạn lệch
        if max_diff > 10.0:
            top_diff_segments.append({
                "vehicle_id": v_id,
                "max_diff": max_diff,
                "time": max_time,
                "raw": max_raw,
                "old_x": max_o,
                "new_x": max_n,
                "diff": max_n - max_o,
            })

        # Tìm các đoạn chữ U (raw tụt sâu rồi hồi phục lại)
        # Tiêu chí chữ U: raw giảm > 15 L rồi tăng lại > 15 L trong vòng <= 30 mẫu
        raw_vals = df_ai["FuelLevel"].to_numpy()
        for i in range(10, len(raw_vals) - 15, 10):
            window = raw_vals[i:i+25]
            if len(window) < 15:
                continue
            drop = np.min(window) - window[0]
            rebound = window[-1] - np.min(window)
            if drop < -15.0 and rebound > 12.0:
                u_shape_segments.append({
                    "vehicle_id": v_id,
                    "time": str(df_ai["FuelTime"].iloc[i]),
                    "start_raw": round(float(window[0]), 1),
                    "min_raw": round(float(np.min(window)), 1),
                    "end_raw": round(float(window[-1]), 1),
                    "drop_liters": round(float(abs(drop)), 1),
                    "rebound_liters": round(float(rebound), 1),
                })
                break

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs("artifacts/fleet_evaluation", exist_ok=True)
    df_summary.to_csv("artifacts/fleet_evaluation/fleet_28_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "="*80)
    print("TỔNG HỢP TOÀN ĐỘI XE (28 XE):")
    print(f"Tổng số xe có thay đổi (max_diff > 0.1 L): {(df_summary['max_diff'] > 0.1).sum()} / {len(df_summary)}")
    print(f"Tổng số xe có thay đổi lớn (max_diff > 10 L): {(df_summary['max_diff'] > 10.0).sum()} / {len(df_summary)}")
    print(f"Tổng số xe khớp tuyệt đối (max_diff < 0.01 L): {(df_summary['max_diff'] < 0.01).sum()} / {len(df_summary)}")
    print("="*80)

    print("\nDANH SÁCH CÁC XE CÓ THAY ĐỔI LỚN NHẤT:")
    df_changed = df_summary[df_summary["max_diff"] > 5.0].sort_values("max_diff", ascending=False)
    for _, r in df_changed.iterrows():
        print(f"Xe {r['vehicle_id']:10s} | MaxDiff: {r['max_diff']:6.2f} L | Time: {r['max_diff_time']} | Raw: {r['max_diff_raw']:6.1f} | Old_x: {r['max_diff_old']:6.2f} -> New_x: {r['max_diff_new']:6.2f} | Refuels Old/New: {r['old_refuel_count']}/{r['new_refuel_count']}")

    print("\nDANH SÁCH XE CÓ ĐOẠN CHỮ U (GIẢM SÂU RỒI HỒI PHỤC):")
    for u in u_shape_segments[:8]:
        print(f"Xe {u['vehicle_id']:10s} | Time: {u['time']} | Raw: {u['start_raw']} -> {u['min_raw']} -> {u['end_raw']} | Tụt: {u['drop_liters']} L, Hồi: {u['rebound_liters']} L")

    # Lưu chi tiết đoạn chữ U
    pd.DataFrame(u_shape_segments).to_csv("artifacts/fleet_evaluation/u_shape_segments.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
