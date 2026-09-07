import glob
import os
import sys
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime as new_filter
from scratch.legacy_adaptive_filter import filter_ai_enhanced_adaptive_realtime as legacy_filter


def main():
    model, metadata = load_fuel_state_classifier("models/fuel_state_classifier")
    files = sorted(glob.glob("TienXuLy/*_processed.csv"))
    vehicle_files = [f for f in files if "Bao_cao_lich_su" not in f]

    truly_unconfirmed = []
    total_refuels = 0

    for fpath in vehicle_files:
        v_id = Path(fpath).name.replace("_processed.csv", "")
        df = pd.read_csv(fpath)
        if len(df) < 50:
            continue
        df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")
        
        trace = []
        clean = np.array(new_filter(df_ai, config={"trace_collector": trace}))
        df_t = pd.DataFrame(trace)
        
        raw = df_ai["FuelLevel"].to_numpy()
        times = pd.to_datetime(df_ai["FuelTime"])
        
        # Tìm các đợt nạp: raw tăng từ mức thấp lên mức cao bền vững
        # Tiêu chí: tìm các vị trí có branch "is_refuel" trong bản cũ (legacy),
        # xem bản mới có xác nhận không, hoặc kiểm tra raw tăng > 30L và duy trì > 30 phút
        trace_old = []
        clean_old = np.array(legacy_filter(df_ai, config={"trace_collector": trace_old}))
        df_old_t = pd.DataFrame(trace_old)
        
        # Lấy các cụm is_refuel của bản cũ
        old_refuel_indices = df_old_t[df_old_t["branch_selected"] == "is_refuel"].index.tolist()
        
        # Nhóm các chỉ số refuel gần nhau thành 1 đợt nạp (trong vòng 30 mẫu)
        clusters = []
        if old_refuel_indices:
            curr_cluster = [old_refuel_indices[0]]
            for idx in old_refuel_indices[1:]:
                if idx - curr_cluster[-1] <= 25:
                    curr_cluster.append(idx)
                else:
                    clusters.append(curr_cluster)
                    curr_cluster = [idx]
            clusters.append(curr_cluster)
            
        for cl in clusters:
            start_idx = cl[0]
            end_idx = cl[-1]
            
            # Kiểm tra xem đợt nạp này có thật không (mức sau đợt nạp có duy trì cao hơn mức trước ít nhất 20L sau 20 mẫu không?)
            post_window = raw[end_idx+5:end_idx+25]
            pre_window = raw[max(0, start_idx-10):start_idx]
            if len(post_window) < 5 or len(pre_window) < 3:
                continue
            
            post_med = float(np.median(post_window))
            pre_med = float(np.median(pre_window))
            gain = post_med - pre_med
            
            if gain >= 20.0:
                total_refuels += 1
                # Kiểm tra xem bản mới có tăng lên mức post_med không:
                clean_after = float(np.median(clean[end_idx+5:end_idx+25]))
                clean_before = float(np.median(clean[max(0, start_idx-10):start_idx]))
                clean_gain = clean_after - clean_before
                
                # Kiểm tra xem trong khoảng [start_idx, end_idx+15] bản mới có kích hoạt is_refuel không
                sub_branches = df_t["branch_selected"].iloc[start_idx:end_idx+20].tolist()
                has_refuel = "is_refuel" in sub_branches
                
                if not has_refuel and clean_gain < 0.6 * gain:
                    truly_unconfirmed.append({
                        "vehicle_id": v_id,
                        "time": str(times.iloc[start_idx]),
                        "gain": round(gain, 1),
                        "clean_gain": round(clean_gain, 1),
                        "pre": round(pre_med, 1),
                        "post": round(post_med, 1),
                        "clean_post": round(clean_after, 1),
                    })

    print(f"Tổng số đợt nạp thực tế (dựa trên cụm nhảy bản cũ có mức duy trì cao thật): {total_refuels}")
    print(f"Số đợt bị bản mới bỏ sót (unconfirmed): {len(truly_unconfirmed)}")
    if truly_unconfirmed:
        for u in truly_unconfirmed:
            print(f"  Xe {u['vehicle_id']} tại {u['time']}: Nạp thật +{u['gain']} L ({u['pre']} -> {u['post']}), nhưng Clean chỉ tăng {u['clean_gain']} L")
    else:
        print("=> 100% CÁC ĐỢT NẠP NHIÊN LIỆU THẬT ĐỀU ĐƯỢC BẢN MỚI XÁC NHẬN THÀNH CÔNG!")


if __name__ == "__main__":
    main()
