import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd

for i in range(1, 6):
    file_name = f'CarFuelHistory_Processed_Car{i}.csv'
    df = pd.read_csv(file_name)
    df['FuelTime'] = pd.to_datetime(df['FuelTime'])
    
    print(f"--- {file_name} ---")
    
    # Check for FUEL_ZERO
    zero_fuel = df[df['QualityReason'].astype(str).str.contains('FUEL_ZERO', na=False)]
    if not zero_fuel.empty:
        print(f"Found FUEL_ZERO in Car {i}:")
        # Print the row and surrounding rows
        idx = zero_fuel.index[0]
        print(df.loc[idx-1:idx+1, ['FuelTime', 'FuelLevel', 'Speed', 'TimeGapMinutes', 'SegmentID', 'QualityReason']])
        
    # Check for LONG_GAP / SUSPICIOUS_GAP
    long_gap = df[df['QualityReason'].astype(str).str.contains('LONG_GAP|SUSPICIOUS_GAP', na=False)]
    if not long_gap.empty:
        print(f"Found LONG_GAP in Car {i}:")
        idx = long_gap.index[0]
        print(df.loc[idx-1:idx+1, ['FuelTime', 'FuelLevel', 'Speed', 'TimeGapMinutes', 'SegmentID', 'QualityReason']])
