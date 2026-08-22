import os
import glob
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

def run_data_profiling(folder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(folder_path, "*_processed.csv"))
    
    stats_list = []
    patterns_list = []
    
    TH_GAP = 15.0
    
    for file in all_files:
        df = pd.read_csv(file)
        if df.empty or 'FuelLevel' not in df.columns:
            continue
            
        vehicle = os.path.basename(file).replace('_processed.csv', '')
        
        # 1. Tính toán Delta
        df['DeltaFuel'] = df['FuelLevel'].diff().fillna(0.0)
        df['DeltaFuel_Next'] = df['DeltaFuel'].shift(-1).fillna(0.0)
        df['AbsDelta'] = df['DeltaFuel'].abs()
        
        # 2. Tính toán thống kê cơ bản (Max, Min, Mean, Std, P95...)
        fuel_min = df['FuelLevel'].min()
        fuel_max = df['FuelLevel'].max()
        fuel_mean = df['FuelLevel'].mean()
        fuel_std = df['FuelLevel'].std()
        
        # Ngưỡng phân loại động theo dung tích (Capacity-aware Thresholds)
        TH_NOISE_NHO = max(1.0, fuel_max * 0.005) # Mức Normal là dưới 0.5% dung tích (Tối thiểu 1 Lít)
        TH_NOISE_MANH = max(3.0, fuel_max * 0.02) # Mức Noise Mạnh/Extreme là trên 2% dung tích (Tối thiểu 3 Lít)
        
        gap_min = df['TimeGapMinutes'].min()
        gap_max = df['TimeGapMinutes'].max()
        gap_mean = df['TimeGapMinutes'].mean()
        gap_median = df['TimeGapMinutes'].median()
        gap_p95 = df['TimeGapMinutes'].quantile(0.95)
        
        delta_max = df['AbsDelta'].max()
        delta_mean = df['AbsDelta'].mean()
        delta_median = df['AbsDelta'].median()
        delta_p95 = df['AbsDelta'].quantile(0.95)
        
        # 3. Phân loại Mẫu (Pattern Classification)
        num_samples = len(df)
        
        c_normal = 0
        c_noise_nho = 0
        c_noise_manh = 0
        c_spike = 0
        c_rise = 0
        c_drop = 0
        c_gap = 0
        
        for i in range(num_samples):
            gap = df['TimeGapMinutes'].iloc[i]
            d = df['DeltaFuel'].iloc[i]
            abs_d = abs(d)
            d_next = df['DeltaFuel_Next'].iloc[i]
            
            if pd.notna(gap) and gap > TH_GAP:
                c_gap += 1
                
            if abs_d >= TH_NOISE_MANH:
                # Kiểm tra Spike (nhảy lên rồi nhảy xuống ngay, ngược dấu và cùng độ lớn tương đối)
                if (d * d_next < 0) and (abs(d_next) >= TH_NOISE_MANH * 0.7):
                    c_spike += 1
                else:
                    if d > 0:
                        c_rise += 1
                    else:
                        c_drop += 1
            elif abs_d >= TH_NOISE_NHO:
                c_noise_manh += 1
            elif abs_d > 0.5:
                c_noise_nho += 1
            else:
                c_normal += 1
                
        # 4. Lưu Stats
        stats_list.append({
            'Vehicle': vehicle,
            'Samples': num_samples,
            'Fuel_Min': round(fuel_min, 1),
            'Fuel_Max': round(fuel_max, 1),
            'Fuel_Mean': round(fuel_mean, 1),
            'Fuel_Std': round(fuel_std, 1),
            'Gap_Min': round(gap_min, 1),
            'Gap_Max': round(gap_max, 1),
            'Gap_Mean': round(gap_mean, 1),
            'Gap_Median': round(gap_median, 1),
            'Gap_P95': round(gap_p95, 1),
            'Delta_Max': round(delta_max, 1),
            'Delta_Mean': round(delta_mean, 2),
            'Delta_Median': round(delta_median, 2),
            'Delta_P95': round(delta_p95, 2),
            'Count_Rise': c_rise,
            'Count_Drop': c_drop,
            'Count_Spike': c_spike
        })
        
        # 5. Lưu Patterns (%)
        patterns_list.append({
            'Vehicle': vehicle,
            'Normal_%': round(c_normal / num_samples * 100, 2),
            'NoiseNhỏ_%': round(c_noise_nho / num_samples * 100, 2),
            'NoiseMạnh_%': round(c_noise_manh / num_samples * 100, 2),
            'Spike_%': round(c_spike / num_samples * 100, 2),
            'Rise_%': round(c_rise / num_samples * 100, 2),
            'Drop_%': round(c_drop / num_samples * 100, 2),
            'Gap_%': round(c_gap / num_samples * 100, 2)
        })
        
    df_stats = pd.DataFrame(stats_list)
    df_patterns = pd.DataFrame(patterns_list)
    
    # Xác định Dominant Pattern
    def get_dominant(row):
        cols = ['Normal_%', 'NoiseNhỏ_%', 'NoiseMạnh_%']
        dom = row[cols].idxmax().replace('_%', '')
        if row['Spike_%'] > 1.0 or row['Rise_%'] > 1.0 or row['Drop_%'] > 1.0:
            return dom + " + Event"
        return dom
        
    df_patterns['Dominant_Pattern'] = df_patterns.apply(get_dominant, axis=1)
    
    df_stats.to_csv(os.path.join(output_dir, 'data_profiling_stats.csv'), index=False)
    df_patterns.to_csv(os.path.join(output_dir, 'data_patterns.csv'), index=False)
    
    print(f"Data Profiling hoàn tất. Đã phân tích {len(stats_list)} xe.")
    print(f"Kết quả lưu tại: {output_dir}")

if __name__ == "__main__":
    run_data_profiling(r'D:\THUCTAP_VICOMSAT\TienXuLy', r'D:\THUCTAP_VICOMSAT\reports')
