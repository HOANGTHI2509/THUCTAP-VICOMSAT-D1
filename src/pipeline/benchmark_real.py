import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D, rts_smooth_1d, is_valid_measurement
from src.pipeline.build_real_dataset import generate_teacher_signal

def calculate_smoothness(series):
    return np.mean(np.abs(np.diff(series.dropna())))

def run_benchmark():
    print("=== FINAL TEST EVALUATION (16 UNSEEN VEHICLES) ===")
    
    csv_files = glob.glob('data/processed/CarFuelHistory_Processed_*_CNN_Realtime.csv')
    
    train_val_cars = ['Car1', 'Car2', 'Car3', 'Car5', '92H-07095']
    
    test_files = [f for f in csv_files if not any(c in f for c in train_val_cars)]
    
    print(f"Found {len(test_files)} unseen vehicles for final test.")
    
    results = []
    all_errors = []
    
    for file in test_files:
        car_id = os.path.basename(file).replace('CarFuelHistory_Processed_', '').replace('_CNN_Realtime.csv', '')
        print(f"\nEvaluating {car_id}...")
        
        df = pd.read_csv(file)
        
        # Original processed file to run Teacher signal
        orig_file = file.replace('_CNN_Realtime.csv', '.csv')
        df_orig = pd.read_csv(orig_file)
        df_orig['FuelTime'] = pd.to_datetime(df_orig['FuelTime'], errors="coerce")
        df_orig['FuelLevel'] = pd.to_numeric(df_orig['FuelLevel'], errors="coerce")
        df_orig['SegmentID'] = pd.to_numeric(df_orig['SegmentID'], errors="coerce")
        df_orig["_OriginalOrder"] = np.arange(len(df_orig))
        df_orig = df_orig.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        # Compute Teacher Signal (Offline Ground Truth Approximation)
        capacity = df_orig['FuelLevel'].quantile(0.99)
        if pd.isna(capacity) or capacity < 50: capacity = 200.0
        
        df_teacher = generate_teacher_signal(df_orig, capacity)
        
        # Merge Teacher signal with CNN output
        df['TeacherFuel'] = df_teacher['TeacherFuel']
        
        # Calculate metrics
        valid_mask = ~df['CNN_Realtime'].isna() & ~df['TeacherFuel'].isna()
        y_true = df.loc[valid_mask, 'TeacherFuel'].values
        y_pred = df.loc[valid_mask, 'CNN_Realtime'].values
        y_raw = df.loc[valid_mask, 'FuelLevel'].values
        
        if len(y_true) < 10:
            continue
            
        rmse_cnn = np.sqrt(np.mean((y_pred - y_true)**2))
        rmse_raw = np.sqrt(np.mean((y_raw - y_true)**2))
        
        errors = np.abs(y_pred - y_true)
        overall_mae = np.mean(errors)
        max_error = np.max(errors)
        
        target_magnitude = np.abs((y_true - y_raw) / capacity)
        event_mask = target_magnitude >= 0.01
        
        if np.sum(event_mask) > 0:
            event_mae = np.mean(errors[event_mask])
            event_rmse = np.sqrt(np.mean((y_pred[event_mask] - y_true[event_mask])**2))
        else:
            event_mae = 0.0
            event_rmse = 0.0
            
        smoothness_cnn = calculate_smoothness(pd.Series(y_pred))
        smoothness_raw = calculate_smoothness(pd.Series(y_raw))
        smoothness_teacher = calculate_smoothness(pd.Series(y_true))
        
        sparsity = np.mean(np.abs(y_pred - y_raw) < 1e-3) * 100.0
        
        # Collect errors for top 10 debug
        df_errors = df.loc[valid_mask, ['FuelTime', 'FuelLevel', 'CNN_Realtime', 'TeacherFuel']].copy()
        df_errors['Error'] = errors
        df_errors['Vehicle'] = car_id
        all_errors.append(df_errors)
        
        results.append({
            'Vehicle': car_id,
            'Samples': len(y_true),
            'Raw_RMSE_to_Teacher': rmse_raw,
            'CNN_RMSE_to_Teacher': rmse_cnn,
            'Overall_MAE': overall_mae,
            'Event_MAE': event_mae,
            'Event_RMSE': event_rmse,
            'Max_Error': max_error,
            'Raw_Smoothness': smoothness_raw,
            'CNN_Smoothness': smoothness_cnn,
            'Teacher_Smoothness': smoothness_teacher,
            'Sparsity': sparsity
        })
        
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS (AVERAGED OVER ALL UNSEEN VEHICLES)")
    print("="*60)
    print(f"Total Test Samples: {df_res['Samples'].sum():,}")
    print(f"Average CNN-GA RMSE vs Teacher: {df_res['CNN_RMSE_to_Teacher'].mean():.2f} Liters")
    print(f"Average Raw RMSE vs Teacher:    {df_res['Raw_RMSE_to_Teacher'].mean():.2f} Liters")
    print(f"Overall MAE:                    {df_res['Overall_MAE'].mean():.2f} Liters")
    print(f"Event MAE (diff >= 1%):         {df_res['Event_MAE'].mean():.2f} Liters")
    print(f"Event RMSE (diff >= 1%):        {df_res['Event_RMSE'].mean():.2f} Liters")
    print(f"Max Error vs Teacher:           {df_res['Max_Error'].max():.2f} Liters")
    print(f"Correction Sparsity (<0.001L):  {df_res['Sparsity'].mean():.2f}%")
    print(f"Average CNN-GA Smoothness:      {df_res['CNN_Smoothness'].mean():.3f} L/step")
    print(f"Average Teacher Smoothness:     {df_res['Teacher_Smoothness'].mean():.3f} L/step")
    print(f"Average Raw Smoothness:         {df_res['Raw_Smoothness'].mean():.3f} L/step")
    
    os.makedirs('artifacts', exist_ok=True)
    df_res.to_csv('artifacts/final_test_benchmark.csv', index=False)
    print("\nSaved detailed benchmark to artifacts/final_test_benchmark.csv")
    
    # Print Top 10 max errors
    df_all_errors = pd.concat(all_errors)
    top_10 = df_all_errors.nlargest(10, 'Error')
    print("\n" + "="*60)
    print("TOP 10 SAMPLES WITH LARGEST ERROR (CNN vs Teacher):")
    print("="*60)
    print(top_10[['Vehicle', 'FuelTime', 'FuelLevel', 'TeacherFuel', 'CNN_Realtime', 'Error']].to_string(index=False))

if __name__ == '__main__':
    run_benchmark()
